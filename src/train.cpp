#include "metrics.h"
#include "model.h"
#include "detector_utils.h"

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
    const auto num_threads = std::thread::hardware_concurrency();
    dlib::command_line_parser parser;
    parser.add_option("architecture", "print the network architecture");
    parser.add_option("name", "name used for sync and net files (default: yolo)", 1);
    parser.add_option("size", "image size for internal usage (default: 512)", 1);
    parser.add_option("test", "visually test with a threshold (default: 0.01)", 1);
    parser.add_option("visualize", "visualize data augmentation instead of training");
    parser.set_group_name("Training Options");
    parser.add_option("batch-gpu", "mini batch size per GPU (default: 8)", 1);
    parser.add_option("warmup", "learning rate warm-up epochs (default: 3)", 1);
    parser.add_option("cosine-epochs", "epochs for the cosine scheduler (default: 0)", 1);
    parser.add_option("gpus", "number of GPUs for the training (default: 1)", 1);
    parser.add_option("iou-ignore", "IoUs above don't incur obj loss (default: 0.5)", 1);
    parser.add_option("iou-anchor", "extra anchors IoU treshold (default: 1)", 1);
    parser.add_option("lambda-obj", "weight for the positive obj class (default: 1)", 1);
    parser.add_option("lambda-box", "weight for the box regression loss (default: 1)", 1);
    parser.add_option("lambda-cls", "weight for the classification loss (default: 1)", 1);
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("min-learning-rate", "minimum learning rate (default: 1e-6)", 1);
    parser.add_option("momentum", "sgd momentum (default: 0.9)", 1);
    parser.add_option("patience", "number of epochs without progress (default: 3)", 1);
    parser.add_option("tune", "path to the network to fine-tune", 1);
    parser.add_option("weight-decay", "sgd weight decay (default: 0.0005)", 1);
    parser.add_option(
        "workers",
        "number data loaders (default: " + std::to_string(num_threads) + ")",
        1);
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
    parser.add_option("solarize", "probability of solarize (default: 0.2)", 1);
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
    parser.check_incompatible_options("patience", "cosine-epochs");
    parser.check_sub_option("crop", "shift");
    const double learning_rate = get_option(parser, "learning-rate", 0.001);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-6);
    const size_t patience = get_option(parser, "patience", 3);
    const size_t cosine_epochs = get_option(parser, "cosine-epochs", 0);
    const double lambda_obj = get_option(parser, "lambda-obj", 1.0);
    const double lambda_box = get_option(parser, "lambda-box", 1.0);
    const double lambda_cls = get_option(parser, "lambda-cls", 1.0);
    const size_t num_gpus = get_option(parser, "gpus", 1);
    const size_t batch_size = get_option(parser, "batch-gpu", 8) * num_gpus;
    const size_t warmup_epochs = get_option(parser, "warmup", 3);
    const size_t image_size = get_option(parser, "size", 512);
    const size_t num_workers = get_option(parser, "workers", num_threads);
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
    const double solarize_prob = get_option(parser, "solarize", 0.2);
    const double iou_ignore_threshold = get_option(parser, "iou-ignore", 0.5);
    const double iou_anchor_threshold = get_option(parser, "iou-anchor", 1.0);
    const float momentum = get_option(parser, "momentum", 0.9);
    const float weight_decay = get_option(parser, "weight-decay", 0.0005);
    const std::string experiment_name = get_option(parser, "name", "yolo");
    const std::string sync_file_name = experiment_name + "_sync";
    const std::string net_file_name = experiment_name + ".dnn";
    const std::string best_metrics_path = experiment_name + "_best_metrics.dat";
    const std::string tune_net_path = get_option(parser, "tune", "");

    const std::string data_path = parser[0];

    dlib::image_dataset_metadata::dataset dataset;
    dlib::image_dataset_metadata::load_image_dataset_metadata(
        dataset,
        data_path + "/training.xml");
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
    options.lambda_obj = lambda_obj;
    options.lambda_box = lambda_box;
    options.lambda_cls = lambda_cls;

    // These are the anchors computed on the COCO dataset, presented in the YOLOv4 paper.
    // options.add_anchors<rgpnet::ytag8>({{12, 16}, {19, 36}, {40, 28}});
    // options.add_anchors<rgpnet::ytag16>({{36, 75}, {76, 55}, {72, 146}});
    // options.add_anchors<rgpnet::ytag32>({{142, 110}, {192, 243}, {459, 401}});
    // These are the anchors computed on the OMNIOUS product_2021-02-25 dataset
    // options.add_anchors<ytag8>({{31, 33}, {62, 42}, {41, 66}});
    // options.add_anchors<ytag16>({{76, 88}, {151, 113}, {97, 184}});
    // options.add_anchors<ytag32>({{205, 243}, {240, 444}, {437, 306}, {430, 549}});
    options.add_anchors<ytag8>({{31, 31}, {47, 51}});
    options.add_anchors<ytag16>({{59, 80}, {100, 90}});
    options.add_anchors<ytag32>({{163, 171}, {209, 316}, {422, 293}, {263, 494}, {469, 534}});

    net_train_type net(options);
    setup_detector(net, options);
    if (parser.option("architecture"))
    {
        rgb_image dummy(image_size, image_size);
        net(dummy);
        std::cerr << net << std::endl;
    }

    if (not tune_net_path.empty())
    {
        // net_train_type pretrained_net;
        dlib::deserialize(tune_net_path) >> net;
        // dlib::layer<57>(net).subnet() = dlib::layer<57>(pretrained_net).subnet();
    }

    // In case we have several GPUs, we can tell the dnn_trainer to make use of them.
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    // We initialize the trainer here, as it will be used in several contexts, depending on the
    // arguments passed the the program.
    auto trainer = dlib::dnn_trainer(net, dlib::sgd(weight_decay, momentum), gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_synchronization_file(sync_file_name, std::chrono::minutes(30));

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
            load_image(image, data_path + "/" + im.filename);
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
    const auto loader = [angle,
                         blur_prob,
                         color_magnitude,
                         color_offset_prob,
                         crop_prob,
                         gamma_magnitude,
                         image_size,
                         mirror_prob,
                         mosaic_prob,
                         perspective_prob,
                         shift,
                         solarize_prob,
                         &data_path,
                         &dataset,
                         &train_data](time_t seed)
    {
        dlib::rand rnd(time(nullptr) + seed);
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
            std::pair<rgb_image, std::vector<dlib::yolo_rect>> result;
            rgb_image image, rotated, blurred, transformed(image_size, image_size);
            const auto idx = rnd.get_random_64bit_number() % dataset.images.size();
            const auto& image_info = dataset.images.at(idx);
            try
            {
                dlib::load_image(image, data_path + "/" + image_info.filename);
            }
            catch (const dlib::image_load_error& e)
            {
                std::cerr << "ERROR: " << e.what() << std::endl;
                result.first.set_size(image_size, image_size);
                dlib::assign_all_pixels(result.first, dlib::rgb_pixel(0, 0, 0));
                result.second = {};
                return result;
            }
            for (const auto& box : image_info.boxes)
                result.second.emplace_back(box.rect, 1, box.label);

            // We alternate between augmenting the full image and random cropping
            if (rnd.get_random_double() < crop_prob)
            {
                std::vector<dlib::yolo_rect> boxes = result.second;
                cropper(image, boxes, result.first, result.second);
            }
            else
            {
                dlib::rectangle_transform tform = rotate_image(
                    image,
                    rotated,
                    rnd.get_double_in_range(-1, 1) * angle * dlib::pi / 180,
                    dlib::interpolate_bilinear());
                for (auto& box : result.second)
                    box.rect = tform(box.rect);

                tform = letterbox_image(rotated, result.first, image_size);
                for (auto& box : result.second)
                    box.rect = tform(box.rect);

                if (rnd.get_random_double() < mirror_prob)
                {
                    tform = flip_image_left_right(result.first);
                    for (auto& box : result.second)
                        box.rect = tform(box.rect);
                }
                if (rnd.get_random_double() < blur_prob)
                {
                    dlib::gaussian_blur(result.first, blurred);
                    result.first = blurred;
                }
                if (rnd.get_random_double() < perspective_prob)
                {
                    const dlib::drectangle r(0, 0, image_size - 1, image_size - 1);
                    std::array<dlib::dpoint, 4> ps{
                        r.tl_corner(),
                        r.tr_corner(),
                        r.bl_corner(),
                        r.br_corner()};
                    const double amount = 0.05;
                    for (auto& corner : ps)
                    {
                        corner.x() += rnd.get_double_in_range(-1, 1) * amount * image_size;
                        corner.y() += rnd.get_double_in_range(-1, 1) * amount * image_size;
                    }
                    const auto ptform = extract_image_4points(result.first, transformed, ps);
                    result.first = transformed;
                    for (auto& box : result.second)
                    {
                        ps[0] = ptform(box.rect.tl_corner());
                        ps[1] = ptform(box.rect.tr_corner());
                        ps[2] = ptform(box.rect.bl_corner());
                        ps[3] = ptform(box.rect.br_corner());
                        const auto lr = std::minmax({ps[0].x(), ps[1].x(), ps[2].x(), ps[3].x()});
                        const auto ud = std::minmax({ps[0].y(), ps[1].y(), ps[2].y(), ps[3].y()});
                        box.rect.left() = lr.first;
                        box.rect.top() = ud.first;
                        box.rect.right() = lr.second;
                        box.rect.bottom() = ud.second;
                    }
                }
            }

            if (rnd.get_random_double() < color_offset_prob)
                dlib::apply_random_color_offset(result.first, rnd);
            else
                disturb_colors(result.first, rnd, gamma_magnitude, color_magnitude);

            if (rnd.get_random_double() < solarize_prob)
            {
                for (auto& p : result.first)
                {
                    if (p.red > 128)
                        p.red = 128 - p.red;
                    if (p.green > 128)
                        p.green = 128 - p.green;
                    if (p.blue > 128)
                        p.blue = 128 - p.blue;
                }
            }

            return result;
        };

        while (train_data.is_enabled())
        {
            if (rnd.get_random_double() < mosaic_prob)
            {
                const double scale = 0.5;
                const long s = image_size * scale;
                std::pair<rgb_image, std::vector<dlib::yolo_rect>> sample;
                sample.first.set_size(image_size, image_size);
                const auto short_dim = cropper.get_min_object_length_short_dim();
                const auto long_dim = cropper.get_min_object_length_long_dim();
                const auto min_coverage = cropper.get_min_object_coverage();
                const std::vector<std::pair<long, long>> pos{{0, 0}, {0, s}, {s, 0}, {s, s}};
                for (const auto& [x, y] : pos)
                {
                    auto tile = get_sample(0);  // do not use random cropping here
                    const dlib::rectangle r(x, y, x + s, y + s);
                    auto si = dlib::sub_image(sample.first, r);
                    resize_image(tile.first, si);
                    for (auto& b : tile.second)
                    {
                        b.rect = translate_rect(scale_rect(b.rect, scale), x, y);
                        // ignore small items
                        if ((b.rect.height() < long_dim and b.rect.width() < long_dim) or
                            (b.rect.height() < short_dim or b.rect.width() < short_dim))
                            b.ignore = true;

                        // ignore items that are not well covered by the current tile
                        const double coverage = b.rect.intersect(r).area() / b.rect.area();
                        if (not b.ignore and coverage < min_coverage)
                            b.ignore = true;

                        sample.second.push_back(std::move(b));
                    }
                }
                train_data.enqueue(sample);
            }
            else
            {
                train_data.enqueue(get_sample(crop_prob));
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
            std::pair<rgb_image, std::vector<dlib::yolo_rect>> sample;
            train_data.dequeue(sample);
            win.clear_overlay();
            win.set_image(sample.first);
            for (const auto& r : sample.second)
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
        std::pair<rgb_image, std::vector<dlib::yolo_rect>> sample;
        while (images.size() < trainer.get_mini_batch_size())
        {
            train_data.dequeue(sample);
            images.push_back(std::move(sample.first));
            bboxes.push_back(std::move(sample.second));
        }
        trainer.train_one_step(images, bboxes);
    };

    const auto num_steps_per_epoch = dataset.images.size() / trainer.get_mini_batch_size();
    const auto warmup_steps = warmup_epochs * num_steps_per_epoch;

    // The training process can be unstable at the beginning.  For this reason, we
    // exponentially increase the learning rate during the first warmup steps.
    if (trainer.get_train_one_step_calls() < warmup_steps)
    {
        if (trainer.get_train_one_step_calls() == 0)
        {
            const dlib::matrix<double> learning_rate_schedule = dlib::linspace(1e-99, learning_rate, warmup_steps);
            trainer.set_learning_rate_schedule(learning_rate_schedule);
            std::cout << "training started with " << warmup_epochs << " warm-up epochs ("
                      << warmup_steps << " steps)" << std::endl;
            std::cout << trainer;
        }
        while (trainer.get_train_one_step_calls() < warmup_steps)
            train();
        trainer.get_net(dlib::force_flush_to_disk::no);
        std::cout << "warm-up finished" << std::endl;
    }

    // setup the trainer after the warm-up
    if (trainer.get_train_one_step_calls() == warmup_steps)
    {
        if (cosine_epochs > 0)
        {
            const size_t cosine_steps = cosine_epochs * num_steps_per_epoch - warmup_steps;
            std::cout << "training with cosine scheduler for " << cosine_epochs - warmup_epochs
                      << " epochs (" << cosine_steps << " steps)" << std::endl;
            // clang-format off
            const dlib::matrix<double> learning_rate_schedule =
            min_learning_rate + 0.5 * (learning_rate - min_learning_rate) *
            (1 + dlib::cos(dlib::linspace(0, cosine_steps, cosine_steps) * dlib::pi / cosine_steps));
            // clang-format on
            trainer.set_learning_rate_schedule(learning_rate_schedule);
        }
        else
        {
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(min_learning_rate);
            trainer.set_learning_rate_shrink_factor(0.1);
            trainer.set_iterations_without_progress_threshold(patience * num_steps_per_epoch);
        }
    }

    trainer.get_net();
    if (trainer.get_train_one_step_calls() == 0)
        std::cout << trainer << std::endl;
    else
        std::cerr << trainer << std::endl;

    double best_map = 0;
    double best_wf1 = 0;
    if (dlib::file_exists(best_metrics_path))
        dlib::deserialize(best_metrics_path) >> best_map >> best_wf1;
    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
    {
        train();
        const auto num_steps = trainer.get_train_one_step_calls();
        if (num_steps % num_steps_per_epoch == 0)
        {
            const auto epoch = num_steps / num_steps_per_epoch;
            net_infer_type tnet(trainer.get_net());
            std::cerr << "computing mean average precison for epoch " << epoch << std::endl;
            dlib::pipe<image_info> test_data(1000);
            test_data_loader test_loader(
                parser[0] + "/testing.xml",
                image_size,
                test_data,
                num_workers);
            std::thread test_loaders([&test_loader]() { test_loader.run(); });
            const auto metrics = compute_metrics(
                tnet,
                test_loader.get_dataset(),
                2 * batch_size / num_gpus,
                test_data,
                0.25,
                std::cerr);

            if (metrics.map > best_map or metrics.weighted_f > best_wf1)
                save_model(net, experiment_name, num_steps, metrics.map, metrics.weighted_f);
            best_map = std::max(metrics.map, best_map);
            best_wf1 = std::max(metrics.weighted_f, best_wf1);

            std::cout << "\n"
                      << "           mAP    mPr    mRc    mF1    µPr    µRc    µF1    wPr    wRc "
                         "   wF1\n";
            std::cout << "EPOCH " << epoch << ": " << std::fixed << std::setprecision(4) << metrics
                      << "\n"
                      << std::endl;

            dlib::serialize(best_metrics_path) << best_map << best_wf1;

            test_data.disable();
            test_loaders.join();
        }
    }

    trainer.get_net();
    std::cout << trainer << std::endl;
    std::cout << "training done" << std::endl;

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
