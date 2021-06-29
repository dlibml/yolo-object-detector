#include "rgpnet.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <tools/imglab/src/metadata_editor.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;

dlib::rectangle_transform
    preprocess_image(const rgb_image& image, rgb_image& output, const long image_size)
{
    return dlib::rectangle_transform(inv(letterbox_image(image, output, image_size)));
}

void postprocess_detections(
    const dlib::rectangle_transform& tform,
    std::vector<dlib::yolo_rect>& detections)
{
    for (auto& d : detections)
        d.rect = tform(d.rect);
}

int main(const int argc, const char** argv)
try
{
    dlib::command_line_parser parser;
    parser.add_option("size", "image size for training (default: 416)", 1);
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("batch-size", "mini batch size (default: 8)", 1);
    parser.add_option("burnin", "learning rate burnin steps (default: 1000)", 1);
    parser.add_option("patience", "number of steps without progress (default: 10000)", 1);
    parser.add_option("workers", "number of worker threads to load data (default: 4)", 1);
    parser.add_option("gpus", "number of GPUs to run the training on (default: 1)", 1);
    parser.add_option("test", "test the detector with a threshold (default: 0.01)", 1);
    parser.add_option("visualize", "visualize data augmentation instead of training");
    parser.add_option("map", "compute the mean average precision");
    parser.add_option("anchors", "compute <arg1> anchor boxes using K-Means", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 || parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        std::cout << "Give the path to a folder containing the training.xml file." << std::endl;
        return 0;
    }
    const double learning_rate = get_option(parser, "learning-rate", 0.001);
    const size_t patience = get_option(parser, "patience", 10000);
    const size_t batch_size = get_option(parser, "batch-size", 8);
    const size_t burnin = get_option(parser, "burnin", 1000);
    const size_t image_size = get_option(parser, "size", 416);
    const size_t num_workers = get_option(parser, "workers", 4);
    const size_t num_gpus = get_option(parser, "gpus", 1);

    const std::string data_directory = parser[0];
    const std::string sync_file_name = "yolov4_sync";

    dlib::image_dataset_metadata::dataset dataset;
    dlib::image_dataset_metadata::load_image_dataset_metadata(
        dataset,
        data_directory + "/training.xml");
    std::cout << "# images: " << dataset.images.size() << std::endl;
    std::map<std::string, size_t> labels;
    size_t num_objects = 0;
    for (const auto& im : dataset.images)
    {
        for (const auto& b : im.boxes)
        {
            labels[b.label]++;
            ++num_objects;
        }
    }
    std::cout << "# labels: " << labels.size() << std::endl;

    dlib::yolo_options options;
    color_mapper string_to_color;
    for (const auto& label : labels)
    {
        std::cout << " - " << label.first << ": " << label.second;
        std::cout << " (" << (100.0 * label.second) / num_objects << "%)\n";
        options.labels.push_back(label.first);
        string_to_color(label.first);
    }

    // If the default anchor boxes don't fit well your data, you should recompute them.
    // Here's an simple way to do it using K-Means clustering.  Note that the approach
    // shown below is suboptimal, since it doesn't group the bounding boxes by size.
    // Grouping the bounding boxes by size and computing the K-Means on each group
    // would make more sense, since each stride of the network is meant to output a
    // boxes at a particular size, but that is very specific to the network architecture
    // and the dataset itself.
    if (parser.option("anchors"))
    {
        const auto num_clusers = std::stoul(parser.option("anchors").argument());
        std::vector<dlib::matrix<double, 2, 1>> samples;
        // First we need to rescale the bounding boxes to match the image size at training time.
        for (const auto& image_info : dataset.images)
        {
            const auto scale = image_size / std::max<double>(image_info.width, image_info.height);
            for (const auto& box : image_info.boxes)
            {
                dlib::matrix<double, 2, 1> sample;
                sample(0) = box.rect.width() * scale;
                sample(1) = box.rect.height() * scale;
                samples.push_back(std::move(sample));
            }
        }
        // Now we can compute K-Means clustering
        randomize_samples(samples);
        std::cout << "Computing anchors for " << samples.size() << " samples" << std::endl;
        std::vector<dlib::matrix<double, 2, 1>> anchors;
        pick_initial_centers(num_clusers, anchors, samples);
        find_clusters_using_kmeans(samples, anchors);
        anchors[0] = 12, 16;
        anchors[1] = 19, 36;
        anchors[2] = 40, 28;
        anchors[3] = 36, 75;
        anchors[4] = 76, 55;
        anchors[5] = 72, 146;
        anchors[6] = 142, 110;
        anchors[7] = 192, 243;
        anchors[8] = 459, 401;
        std::sort(
            anchors.begin(),
            anchors.end(),
            [](const auto& a, const auto& b) { return a(0) * a(1) < b(0) * b(1); });
        for (const auto& c : anchors)
            std::cout << round(c(0)) << 'x' << round(c(1)) << std::endl;
        // And check the average IoU of the newly computed anchor boxes and the training samples.
        double average_iou = 0;
        for (const auto& s : samples)
        {
            const auto sample = dlib::centered_drect(dlib::dpoint(0, 0), s(0), s(1));
            double best_iou = 0;
            for (const auto& a : anchors)
            {
                const auto anchor = centered_drect(dlib::dpoint(0, 0), a(0), a(1));
                best_iou = std::max(best_iou, box_intersection_over_union(sample, anchor));
            }
            average_iou += best_iou;
        }
        std::cout << "Average IoU: " << average_iou / samples.size() << std::endl;
        return EXIT_SUCCESS;
    }

    // When computing the objectness loss in YOLO, predictions that do not have an IoU
    // with any ground truth box of at least options.iou_ignore_threshold, will be
    // treated as not capable of detecting an object, an therefore incur loss.
    // Predictions above this threshold will be ignored, i.e. will not contribute to the
    // loss. Good values are 0.7 or 0.5.
    options.iou_ignore_threshold = 0.5;
    // These are the anchors computed on COCO dataset, presented in the YOLOv3 paper.
    options.add_anchors<rgpnet::ytag8>({{10, 13}, {16, 30}, {33, 23}});
    options.add_anchors<rgpnet::ytag16>({{30, 61}, {62, 45}, {59, 119}});
    options.add_anchors<rgpnet::ytag32>({{116, 90}, {156, 198}, {373, 326}});
    rgpnet::train net(options);

    // The training process can be unstable at the beginning.  For this reason, we exponentially
    // increase the learning rate during the first burnin steps.
    const dlib::matrix<double> learning_rate_schedule =
        learning_rate * pow(dlib::linspace(1e-12, 1, burnin), 4);

    // In case we have several GPUs, we can tell the dnn_trainer to make use of them.
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    // We initialize the trainer here, as it will be used in several contexts, depending on the
    // arguments passed the the program.
    dlib::dnn_trainer<decltype(net)> trainer(net, dlib::sgd(0.0005, 0.9), gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_learning_rate_schedule(learning_rate_schedule);
    trainer.set_synchronization_file(sync_file_name, std::chrono::minutes(15));
    std::cout << trainer;

    // If the training has started and a synchronization file has already been saved to disk,
    // we can re-run this program with the --test option and a confidence threshold to see
    // how the training is going.
    if (parser.option("test"))
    {
        if (!dlib::file_exists(sync_file_name))
        {
            std::cout << "Could not find file " << sync_file_name << std::endl;
            return EXIT_FAILURE;
        }
        const double threshold = get_option(parser, "test", 0.01);
        dlib::image_window win;
        rgb_image image, resized;
        for (const auto& im : dataset.images)
        {
            win.clear_overlay();
            load_image(image, data_directory + "/" + im.filename);
            win.set_title(im.filename);
            win.set_image(image);
            const auto tform = preprocess_image(image, resized, image_size);
            auto detections = net.process(resized, threshold);
            postprocess_detections(tform, detections);
            std::cout << "# detections: " << detections.size() << std::endl;
            for (const auto& det : detections)
            {
                win.add_overlay(det.rect, string_to_color(det.label), det.label);
                std::cout << det.label << ": " << det.rect << " " << det.detection_confidence
                          << std::endl;
            }
            std::cin.get();
        }
        return EXIT_SUCCESS;
    }

    // If the training has started and a synchronization file has already been saved to disk,
    // we can re-run this program with the --map option to compute the mean average precision
    // on the test set.
    if (parser.option("map"))
    {
        dlib::image_dataset_metadata::dataset dataset;
        dlib::image_dataset_metadata::load_image_dataset_metadata(
            dataset,
            data_directory + "/testing.xml");
        if (!dlib::file_exists(sync_file_name))
        {
            std::cout << "Could not find file " << sync_file_name << std::endl;
            return EXIT_FAILURE;
        }
        rgb_image image, resized;
        std::map<std::string, std::vector<std::pair<double, bool>>> hits;
        std::map<std::string, unsigned long> missing;
        for (const auto& label : options.labels)
        {
            hits[label] = std::vector<std::pair<double, bool>>();
            missing[label] = 0;
        }
        std::cout << "computing mean average precision for " << dataset.images.size()
                  << " images..." << std::endl;
        for (size_t i = 0; i < dataset.images.size(); ++i)
        {
            const auto& im = dataset.images[i];
            load_image(image, data_directory + "/" + im.filename);
            const auto tform = preprocess_image(image, resized, image_size);
            auto dets = net.process(resized, 0.005);
            postprocess_detections(tform, dets);
            std::vector<bool> used(dets.size(), false);
            // true positives: truths matched by detections
            for (size_t t = 0; t < im.boxes.size(); ++t)
            {
                bool found_match = false;
                for (size_t d = 0; d < dets.size(); ++d)
                {
                    if (used[d])
                        continue;
                    if (im.boxes[t].label == dets[d].label &&
                        box_intersection_over_union(
                            dlib::drectangle(im.boxes[t].rect),
                            dets[d].rect) > 0.5)
                    {
                        used[d] = true;
                        found_match = true;
                        hits.at(dets[d].label).emplace_back(dets[d].detection_confidence, true);
                        break;
                    }
                }
                // false negatives: truths not matched
                if (!found_match)
                    missing.at(im.boxes[t].label)++;
            }
            // false positives: detections not matched
            for (size_t d = 0; d < dets.size(); ++d)
            {
                if (!used[d])
                    hits.at(dets[d].label).emplace_back(dets[d].detection_confidence, false);
            }
            std::cout << "progress: " << i << '/' << dataset.images.size() << "\t\t\t\r"
                      << std::flush;
        }
        double map = 0;
        for (auto& item : hits)
        {
            std::sort(item.second.rbegin(), item.second.rend());
            const double ap = dlib::average_precision(item.second, missing[item.first]);
            std::cout << dlib::rpad(item.first + ": ", 16, " ") << ap * 100 << '%' << std::endl;
            map += ap;
        }
        std::cout << dlib::rpad(std::string("mAP: "), 16, " ") << map / hits.size() * 100 << '%'
                  << std::endl;
        return EXIT_SUCCESS;
    }

    // Create some data loaders which will load the data, and perform som data augmentation.
    dlib::pipe<std::pair<rgb_image, std::vector<dlib::yolo_rect>>> train_data(1000);
    const auto loader = [&dataset, &data_directory, &train_data, &image_size](time_t seed)
    {
        dlib::rand rnd(time(nullptr) + seed);
        rgb_image image, rotated;
        std::pair<rgb_image, std::vector<dlib::yolo_rect>> temp;
        dlib::random_cropper cropper;
        cropper.set_seed(time(nullptr) + seed);
        cropper.set_chip_dims(image_size, image_size);
        cropper.set_max_object_size(0.9);
        cropper.set_min_object_size(10, 10);
        cropper.set_max_rotation_degrees(10);
        cropper.set_translate_amount(0.5);
        cropper.set_randomly_flip(true);
        cropper.set_background_crops_fraction(0);
        while (train_data.is_enabled())
        {
            const auto idx = rnd.get_random_32bit_number() % dataset.images.size();
            load_image(image, data_directory + "/" + dataset.images[idx].filename);
            for (const auto& box : dataset.images[idx].boxes)
                temp.second.emplace_back(box.rect, 1, box.label);

            // We alternate between augmenting the full image and random cropping
            if (rnd.get_random_double() > 0.5)
            {
                dlib::rectangle_transform tform = rotate_image(
                    image,
                    rotated,
                    rnd.get_double_in_range(-5 * dlib::pi / 180, 5 * dlib::pi / 180),
                    dlib::interpolate_bilinear());
                for (auto& box : temp.second)
                    box.rect = tform(box.rect);

                tform = letterbox_image(rotated, temp.first, image_size);
                for (auto& box : temp.second)
                    box.rect = tform(box.rect);

                if (rnd.get_random_double() > 0.5)
                {
                    tform = flip_image_left_right(temp.first);
                    for (auto& box : temp.second)
                        box.rect = tform(box.rect);
                }
            }
            else
            {
                std::vector<dlib::yolo_rect> boxes = temp.second;
                cropper(image, boxes, temp.first, temp.second);
            }
            disturb_colors(temp.first, rnd);
            train_data.enqueue(temp);
        }
    };

    std::vector<std::thread> data_loaders;
    for (size_t i = 0; i < num_workers; ++i)
        data_loaders.emplace_back([loader, i]() { loader(i + 1); });

    // It is always a good idea to visualize the training samples.  By passing the --visualize
    // flag, we can see the training samples that will be fed to the dnn_trainer.
    if (parser.option("visualize"))
    {
        dlib::image_window win;
        while (true)
        {
            std::pair<rgb_image, std::vector<dlib::yolo_rect>> temp;
            train_data.dequeue(temp);
            win.clear_overlay();
            win.set_image(temp.first);
            for (const auto& r : temp.second)
            {
                auto color = string_to_color(r.label);
                // make semi-transparent and cross-out the ignored boxes
                if (r.ignore)
                {
                    color.alpha = 128;
                    win.add_overlay(r.rect.tl_corner(), r.rect.br_corner(), color);
                    win.add_overlay(r.rect.tr_corner(), r.rect.bl_corner(), color);
                }
                win.add_overlay(r.rect, color, r.label);
            }
            std::cout << "Press enter to visualize the next training sample.";
            std::cin.get();
        }
    }

    std::vector<rgb_image> images;
    std::vector<std::vector<dlib::yolo_rect>> bboxes;

    // The main training loop, that we will reuse for the warmup and the rest of the training.
    const auto train = [&images, &bboxes, &train_data, &trainer]()
    {
        images.clear();
        bboxes.clear();
        std::pair<rgb_image, std::vector<dlib::yolo_rect>> temp;
        while (images.size() < trainer.get_mini_batch_size())
        {
            train_data.dequeue(temp);
            images.push_back(std::move(temp.first));
            bboxes.push_back(move(temp.second));
        }
        trainer.train_one_step(images, bboxes);
    };

    std::cout << "training started with " << burnin << " burn-in steps" << std::endl;
    while (trainer.get_train_one_step_calls() < burnin)
        train();

    std::cout << "burn-in finished" << std::endl;
    trainer.get_net();
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(learning_rate * 1e-3);
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_iterations_without_progress_threshold(patience);
    std::cout << trainer << std::endl;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        train();

    std::cout << "training done" << std::endl;

    trainer.get_net();
    train_data.disable();
    for (auto& worker : data_loaders)
        worker.join();

    dlib::serialize("yolov3.dnn") << net;
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
