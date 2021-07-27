#include "model.h"
#include "utils.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <tools/imglab/src/metadata_editor.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;

int main(const int argc, const char** argv)
try
{
    dlib::command_line_parser parser;
    parser.add_option("architecture", "print the network architecture");
    parser.add_option("name", "name used for sync and net files (default: yolo)", 1);
    parser.add_option("size", "image size for internal usage (default: 512)", 1);
    parser.add_option("test", "visually test with a threshold (default: 0.01)", 1);
    parser.add_option("visualize", "visualize data augmentation instead of training");
    parser.set_group_name("Training Options");
    parser.add_option("batch", "mini batch size (default: 8)", 1);
    parser.add_option("burnin", "learning rate burn-in steps (default: 1000)", 1);
    parser.add_option("gpus", "number of GPUs for the training (default: 1)", 1);
    parser.add_option("iou-ignore", "IoUs above don't incur obj loss (default: 0.5)", 1);
    parser.add_option("iou-anchor", "extra anchors IoU treshold (default: 1)", 1);
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("min-learning-rate", "minimum learning rate (default: 1e-6)", 1);
    parser.add_option("momentum", "sgd momentum (default: 0.9)", 1);
    parser.add_option("patience", "number of steps without progress (default: 10000)", 1);
    parser.add_option("weight-decay", "sgd weight decay (default: 0.0005)", 1);
    parser.add_option("workers", "number of worker data loader threads (default: 4)", 1);
    parser.set_group_name("Data Augmentation Options");
    parser.add_option("angle", "max random rotation in degrees (default: 5)", 1);
    parser.add_option("blur", "probability of blurring the image (default: 0.5)", 1);
    parser.add_option("color", "color magnitude (default: 0.2)", 1);
    parser.add_option("color-offset", "random color offset probability (default: 0.5)", 1);
    parser.add_option("crop", "no-mosaic random crop probability (default: 0.5)", 1);
    parser.add_option("gamma", "gamma magnitude (default: 0.5)", 1);
    parser.add_option("mirror", "mirror probability (default: 0.5)", 1);
    parser.add_option("mosaic", "mosaic probability (default: 0.5)", 1);
    parser.add_option("perspective", "perspective probability (default: 0.5)", 1);
    parser.add_option("shift", "crop shift relative to box size (default: 0.5)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 || parser.option("h") || parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/DIRECTORY" << std::endl;
        parser.print_options();
        std::cout << "Give the path to a folder containing the training.xml file." << std::endl;
        return 0;
    }
    parser.check_option_arg_range<double>("iou-ignore", 0, 1);
    parser.check_option_arg_range<double>("iou-anchor", 0, 1);
    parser.check_option_arg_range<double>("mirror", 0, 1);
    parser.check_option_arg_range<double>("mosaic", 0, 1);
    parser.check_option_arg_range<double>("crop", 0, 1);
    parser.check_option_arg_range<double>("perspective", 0, 1);
    parser.check_option_arg_range<double>("color-offset", 0, 1);
    parser.check_option_arg_range<double>("gamma", 0, std::numeric_limits<double>::max());
    parser.check_option_arg_range<double>("color", 0, 1);
    parser.check_option_arg_range<double>("blur", 0, 1);
    parser.check_sub_option("crop", "shift");
    const double learning_rate = get_option(parser, "learning-rate", 0.001);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-6);
    const size_t patience = get_option(parser, "patience", 10000);
    const size_t batch_size = get_option(parser, "batch", 8);
    const size_t burnin = get_option(parser, "burnin", 1000);
    const size_t image_size = get_option(parser, "size", 512);
    const size_t num_workers = get_option(parser, "workers", 4);
    const size_t num_gpus = get_option(parser, "gpus", 1);
    const double mirror_prob = get_option(parser, "mirror", 0.5);
    const double mosaic_prob = get_option(parser, "mosaic", 0.5);
    const double crop_prob = get_option(parser, "crop", 0.5);
    const double blur_prob = get_option(parser, "blur", 0.5);
    const double perspective_prob = get_option(parser, "perspective", 0.5);
    const double color_offset_prob = get_option(parser, "color-offset", 0.5);
    const double gamma_magnitude = get_option(parser, "gamma", 0.5);
    const double color_magnitude = get_option(parser, "color", 0.2);
    const double angle = get_option(parser, "angle", 5);
    const double shift = get_option(parser, "shift", 0.5);
    const double iou_ignore_threshold = get_option(parser, "iou-ignore", 0.5);
    const double iou_anchor_threshold = get_option(parser, "iou-anchor", 1.0);
    const float momentum = get_option(parser, "momentum", 0.9);
    const float weight_decay = get_option(parser, "weight-decay", 0.0005);
    const std::string experiment_name = get_option(parser, "name", "yolo");
    const std::string sync_file_name = experiment_name + "_sync";
    const std::string net_file_name = experiment_name + ".dnn";

    const std::string data_directory = parser[0];

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
    options.iou_ignore_threshold = iou_ignore_threshold;
    options.iou_anchor_threshold = iou_anchor_threshold;

    // These are the anchors computed on the COCO dataset, presented in the YOLOv4 paper.
    // options.add_anchors<rgpnet::ytag8>({{12, 16}, {19, 36}, {40, 28}});
    // options.add_anchors<rgpnet::ytag16>({{36, 75}, {76, 55}, {72, 146}});
    // options.add_anchors<rgpnet::ytag32>({{142, 110}, {192, 243}, {459, 401}});
    // These are the anchors computed on the OMNIOUS product_2021-02-25 dataset
    options.add_anchors<ytag8>({{31, 33}, {62, 42}, {41, 66}});
    options.add_anchors<ytag16>({{76, 88}, {151, 113}, {97, 184}});
    options.add_anchors<ytag32>({{205, 243}, {240, 444}, {437, 306}, {430, 549}});

    model_train model(options);
    auto& net = model.net;
    setup_detector(net, options);
    if (parser.option("architecture"))
        std::cerr << net << std::endl;

    // The training process can be unstable at the beginning.  For this reason, we exponentially
    // increase the learning rate during the first burnin steps.
    const dlib::matrix<double> learning_rate_schedule =
        learning_rate * pow(dlib::linspace(1e-12, 1, burnin), 4);

    // In case we have several GPUs, we can tell the dnn_trainer to make use of them.
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    // We initialize the trainer here, as it will be used in several contexts, depending on the
    // arguments passed the the program.
    auto trainer = model.get_trainer(weight_decay, momentum, gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_learning_rate_schedule(learning_rate_schedule);
    trainer.set_synchronization_file(sync_file_name, std::chrono::minutes(30));
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

    // Create some data loaders which will load the data, and perform som data augmentation.
    dlib::pipe<std::pair<rgb_image, std::vector<dlib::yolo_rect>>> train_data(100 * batch_size);
    const auto loader = [&](time_t seed)
    {
        dlib::rand rnd(time(nullptr) + seed);
        rgb_image image, rotated, blurred;
        dlib::random_cropper cropper;
        cropper.set_seed(time(nullptr) + seed);
        cropper.set_chip_dims(image_size, image_size);
        cropper.set_max_object_size(0.9);
        cropper.set_min_object_size(24, 24);
        cropper.set_min_object_coverage(0.7);
        cropper.set_max_rotation_degrees(angle);
        cropper.set_translate_amount(shift);
        if (mirror_prob == 0)
            cropper.set_randomly_flip(false);
        cropper.set_background_crops_fraction(0);

        const auto get_sample = [&](const double crop_prob = 0.5)
        {
            std::pair<rgb_image, std::vector<dlib::yolo_rect>> sample;
            const auto idx = rnd.get_random_32bit_number() % dataset.images.size();
            try
            {
                dlib::load_image(image, data_directory + "/" + dataset.images[idx].filename);
            }
            catch (const dlib::image_load_error& e)
            {
                std::cerr << "ERROR loading image"
                          << data_directory + "/" + dataset.images[idx].filename << std::endl;
                std::cerr << e.what() << std::endl;
                auto empty = rgb_image(image_size, image_size);
                dlib::assign_all_pixels(empty, dlib::rgb_pixel(0, 0, 0));
                sample.second = {};
                return sample;
            }
            for (const auto& box : dataset.images[idx].boxes)
                sample.second.emplace_back(box.rect, 1, box.label);

            // We alternate between augmenting the full image and random cropping
            if (rnd.get_random_double() > crop_prob)
            {
                dlib::rectangle_transform tform = rotate_image(
                    image,
                    rotated,
                    rnd.get_double_in_range(-angle * dlib::pi / 180, angle * dlib::pi / 180),
                    dlib::interpolate_bilinear());
                for (auto& box : sample.second)
                    box.rect = tform(box.rect);

                tform = letterbox_image(rotated, sample.first, image_size);
                for (auto& box : sample.second)
                    box.rect = tform(box.rect);

                if (rnd.get_random_double() < mirror_prob)
                {
                    tform = flip_image_left_right(sample.first);
                    for (auto& box : sample.second)
                        box.rect = tform(box.rect);
                }
                if (rnd.get_random_double() < blur_prob)
                {
                    dlib::gaussian_blur(sample.first, blurred);
                    std::swap(sample.first, blurred);
                }
                if (rnd.get_random_double() < perspective_prob)
                {
                    image = sample.first;
                    const dlib::drectangle r(0, 0, image.nc() - 1, image.nr() - 1);
                    std::array<dlib::dpoint, 4> corners{
                        r.tl_corner(),
                        r.tr_corner(),
                        r.bl_corner(),
                        r.br_corner()};
                    const double amount = image_size / 4.;
                    for (auto& corner : corners)
                    {
                        corner.x() += rnd.get_double_in_range(-amount / 2., amount / 2.);
                        corner.y() += rnd.get_double_in_range(-amount / 2., amount / 2.);
                    }
                    const auto ptform = extract_image_4points(image, sample.first, corners);
                    for (auto& box : sample.second)
                    {
                        corners[0] = ptform(box.rect.tl_corner());
                        corners[1] = ptform(box.rect.tr_corner());
                        corners[2] = ptform(box.rect.bl_corner());
                        corners[3] = ptform(box.rect.br_corner());
                        box.rect.left() = std::min(
                            {corners[0].x(), corners[1].x(), corners[2].x(), corners[3].x()});
                        box.rect.top() = std::min(
                            {corners[0].y(), corners[1].y(), corners[2].y(), corners[3].y()});
                        box.rect.right() = std::max(
                            {corners[0].x(), corners[1].x(), corners[2].x(), corners[3].x()});
                        box.rect.bottom() = std::max(
                            {corners[0].y(), corners[1].y(), corners[2].y(), corners[3].y()});
                    }
                }
            }
            else
            {
                std::vector<dlib::yolo_rect> boxes = sample.second;
                cropper(image, boxes, sample.first, sample.second);
            }

            if (rnd.get_random_double() > color_offset_prob)
                disturb_colors(sample.first, rnd, gamma_magnitude, color_magnitude);
            else
                dlib::apply_random_color_offset(sample.first, rnd);

            return sample;
        };

        while (train_data.is_enabled())
        {
            if (rnd.get_random_double() > mosaic_prob)
            {
                train_data.enqueue(get_sample(crop_prob));
            }
            else
            {
                const double scale = 0.5;
                const long tile_size = image_size * scale;
                std::pair<rgb_image, std::vector<dlib::yolo_rect>> sample;
                sample.first.set_size(image_size, image_size);
                const auto short_dim = cropper.get_min_object_length_short_dim();
                const auto long_dim = cropper.get_min_object_length_long_dim();
                for (size_t i = 0; i < 4; ++i)
                {
                    long x = 0, y = 0;
                    switch (i)
                    {
                    case 0:
                        x = 0 * tile_size;
                        y = 0 * tile_size;
                        break;
                    case 1:
                        x = 0 * tile_size;
                        y = 1 * tile_size;
                        break;
                    case 2:
                        x = 1 * tile_size;
                        y = 0 * tile_size;
                        break;
                    case 3:
                        x = 1 * tile_size;
                        y = 1 * tile_size;
                        break;
                    default:
                        DLIB_CASSERT(false, "Something went terribly wrong");
                    }

                    auto tile = get_sample(0);  // do not use random cropping here
                    const dlib::rectangle r(x, y, x + tile_size, y + tile_size);
                    auto si = dlib::sub_image(sample.first, r);
                    resize_image(tile.first, si);
                    for (auto& b : tile.second)
                    {
                        b.rect = translate_rect(scale_rect(b.rect, scale), x, y);
                        if ((b.rect.height() < long_dim and b.rect.width() < long_dim) or
                            (b.rect.height() < short_dim or b.rect.width() < short_dim))
                            b.ignore = true;
                        sample.second.push_back(std::move(b));
                    }
                }
                train_data.enqueue(sample);
            }
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

    if (trainer.get_train_one_step_calls() < burnin)
    {
        std::cout << "training started with " << burnin << " burn-in steps" << std::endl;
        while (trainer.get_train_one_step_calls() < burnin)
            train();
        std::cout << "burn-in finished" << std::endl;
        trainer.get_net();
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_iterations_without_progress_threshold(patience);
        std::cout << trainer << std::endl;
    }

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
        train();

    std::cout << "training done" << std::endl;

    trainer.get_net();
    train_data.disable();
    for (auto& worker : data_loaders)
        worker.join();

    dlib::serialize(experiment_name + ".dnn") << net;
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
