#include "detector_utils.h"
#include "metrics.h"
#include "model.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <tools/imglab/src/metadata_editor.h>

using namespace dlib;

using rgb_image = matrix<rgb_pixel>;

int main(const int argc, const char** argv)
try
{
    const auto num_threads = std::thread::hardware_concurrency();
    const auto num_threads_str = std::to_string(num_threads);
    command_line_parser parser;
    parser.add_option("architecture", "print the network architecture");
    parser.add_option("name", "name used for sync and net files (default: yolo)", 1);
    parser.add_option("size", "image size for internal usage (default: 512)", 1);
    parser.add_option("test", "visually test with a threshold (default: 0.01)", 1);
    parser.add_option("visualize", "visualize data augmentation instead of training");

    parser.set_group_name("Training Options");
    parser.add_option("batch-gpu", "mini batch size per GPU (default: 8)", 1);
    parser.add_option("warmup", "linear warm-up epochs unless (default: 0)", 1);
    parser.add_option("burnin", "use exponential burn-in instead of linear warm-up");
    parser.add_option("cosine-epochs", "epochs for the cosine scheduler (default: 0)", 1);
    parser.add_option("gpus", "number of GPUs for the training (default: 1)", 1);
    parser.add_option("iou-ignore", "IoUs above don't incur obj loss (default: 0.5)", 1);
    parser.add_option("iou-anchor", "extra anchors IoU treshold (default: 1)", 1);
    parser.add_option("lambda-obj", "weight for the objectness loss  (default: 1)", 1);
    parser.add_option("lambda-box", "weight for the box regression loss (default: 1)", 1);
    parser.add_option("lambda-cls", "weight for the classification loss (default: 1)", 1);
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("min-learning-rate", "minimum learning rate (default: 1e-6)", 1);
    parser.add_option("momentum", "sgd momentum (default: 0.9)", 1);
    parser.add_option("patience", "number of epochs without progress (default: 3)", 1);
    parser.add_option("test-period", "test a batch every <arg> steps (default: 0)", 1);
    parser.add_option("tune", "path to the network to fine-tune", 1);
    parser.add_option("weight-decay", "sgd weight decay (default: 0.0005)", 1);
    parser.add_option("workers", "number data loaders (default: " + num_threads_str + ")", 1);

    parser.set_group_name("Data Augmentation Options");
    parser.add_option("angle", "max random rotation in degrees (default: 3)", 1);
    parser.add_option("blur", "probability of blurring the image (default: 0.0)", 1);
    parser.add_option("color", "color magnitude (default: 0.5)", 1);
    parser.add_option("color-jitter", "random color jitter probability (default: 0.5)", 1);
    parser.add_option("gamma", "gamma magnitude (default: 1.5)", 1);
    parser.add_option("ignore-partial", "ignore intead of removing partial objects");
    parser.add_option("min-coverage", "remove objects partially covered (default: 0.75)", 1);
    parser.add_option("mirror", "mirror probability (default: 0.5)", 1);
    parser.add_option("mixup", "mixup probability (default: 0)", 1);
    parser.add_option("mosaic", "mosaic probability (default: 0.5)", 1);
    parser.add_option("perspective", "perspective probability (default: 0.2)", 1);
    parser.add_option("scale", "random scale gain (default: 0.5)", 1);
    parser.add_option("shift", "random shift fraction (default: 0.2)", 1);
    parser.add_option("solarize", "probability of solarize (default: 0.0)", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");

    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/DIRECTORY\n";
        parser.print_options();
        std::cout << "Give the path to a folder containing the training.xml|testing.xml files.\n";
        return EXIT_SUCCESS;
    }
    parser.check_option_arg_range<double>("iou-ignore", 0, 1);
    parser.check_option_arg_range<double>("iou-anchor", 0, 1);
    parser.check_option_arg_range<double>("mirror", 0, 1);
    parser.check_option_arg_range<double>("mixup", 0, 1);
    parser.check_option_arg_range<double>("mosaic", 0, 1);
    parser.check_option_arg_range<double>("scale", 0, 1);
    parser.check_option_arg_range<double>("perspective", 0, 1);
    parser.check_option_arg_range<double>("min-coverage", 0, 1);
    parser.check_option_arg_range<double>("color-jitter", 0, 1);
    parser.check_option_arg_range<double>("gamma", 0, std::numeric_limits<double>::max());
    parser.check_option_arg_range<double>("color", 0, 1);
    parser.check_option_arg_range<double>("blur", 0, 1);
    parser.check_incompatible_options("patience", "cosine-epochs");
    parser.check_sub_option("warmup", "burnin");
    const double learning_rate = get_option(parser, "learning-rate", 0.001);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-6);
    const size_t patience = get_option(parser, "patience", 3);
    const double cosine_epochs = get_option(parser, "cosine-epochs", 0.0);
    const double lambda_obj = get_option(parser, "lambda-obj", 1.0);
    const double lambda_box = get_option(parser, "lambda-box", 1.0);
    const double lambda_cls = get_option(parser, "lambda-cls", 1.0);
    const size_t num_gpus = get_option(parser, "gpus", 1);
    const size_t batch_size = get_option(parser, "batch-gpu", 8) * num_gpus;
    const double warmup_epochs = get_option(parser, "warmup", 0.0);
    const bool burnin = parser.option("burnin");
    const size_t test_period = get_option(parser, "test-period", 0);
    const size_t image_size = get_option(parser, "size", 512);
    const size_t num_workers = get_option(parser, "workers", num_threads);
    const double mirror_prob = get_option(parser, "mirror", 0.5);
    const double mosaic_prob = get_option(parser, "mosaic", 0.5);
    const double mixup_prob = get_option(parser, "mixup", 0.0);
    const double blur_prob = get_option(parser, "blur", 0.0);
    const double perspective_prob = get_option(parser, "perspective", 0.2);
    const double color_jitter_prob = get_option(parser, "color-jitter", 0.5);
    const double gamma_magnitude = get_option(parser, "gamma", 1.5);
    const double color_magnitude = get_option(parser, "color", 0.5);
    const double angle = get_option(parser, "angle", 3);
    const double scale_gain = get_option(parser, "scale", 0.5);
    const double shift_frac = get_option(parser, "shift", 0.2);
    const double min_coverage = get_option(parser, "min-coverage", 0.75);
    const bool ignore_partial_boxes = parser.option("ignore-partial");
    const double solarize_prob = get_option(parser, "solarize", 0.0);
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

    image_dataset_metadata::dataset train_dataset;
    image_dataset_metadata::load_image_dataset_metadata(
        train_dataset,
        data_path + "/training.xml");
    std::clog << "# train images: " << train_dataset.images.size() << '\n';
    std::map<std::string, size_t> labels;
    size_t num_objects = 0;
    for (const auto& im : train_dataset.images)
    {
        for (const auto& b : im.boxes)
        {
            labels[b.label]++;
            ++num_objects;
        }
    }
    std::clog << "# labels: " << labels.size() << '\n';

    yolo_options options;
    color_mapper string_to_color;
    for (const auto& label : labels)
    {
        std::clog << " - " << label.first << ": " << label.second;
        std::clog << " (" << (100.0 * label.second) / num_objects << "%)\n";
        options.labels.push_back(label.first);
        string_to_color(label.first);
    }
    options.iou_ignore_threshold = iou_ignore_threshold;
    options.iou_anchor_threshold = iou_anchor_threshold;
    options.lambda_obj = lambda_obj;
    options.lambda_box = lambda_box;
    options.lambda_cls = lambda_cls;

    image_dataset_metadata::dataset test_dataset;
    image_dataset_metadata::load_image_dataset_metadata(test_dataset, data_path + "/testing.xml");
    std::clog << "# test images: " << test_dataset.images.size() << '\n';

    // These are the anchors computed on the COCO dataset, presented in the YOLOv4 paper.
    // options.add_anchors<rgpnet::ytag8>({{12, 16}, {19, 36}, {40, 28}});
    // options.add_anchors<rgpnet::ytag16>({{36, 75}, {76, 55}, {72, 146}});
    // options.add_anchors<rgpnet::ytag32>({{142, 110}, {192, 243}, {459, 401}});
    // These are the anchors computed on the OMNIOUS product_2021-02-25 dataset
    // options.add_anchors<ytag8>({{31, 33}, {62, 42}, {41, 66}});
    // options.add_anchors<ytag16>({{76, 88}, {151, 113}, {97, 184}});
    // options.add_anchors<ytag32>({{205, 243}, {240, 444}, {437, 306}, {430, 549}});
    // options.add_anchors<ytag8>({{31, 31}, {47, 51}});
    // options.add_anchors<ytag16>({{59, 80}, {100, 90}});
    // options.add_anchors<ytag32>({{163, 171}, {209, 316}, {422, 293}, {263, 494}, {469, 534}});
    options.add_anchors<ytag3>({{30, 29}, {38, 52}, {54, 47}});
    options.add_anchors<ytag4>({{53, 88}, {85, 59}, {99, 103}});
    options.add_anchors<ytag5>({{105, 181}, {170, 121}, {197, 211}});
    options.add_anchors<ytag6>({{193, 329}, {365, 258}, {268, 493}, {469, 483}});

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
        deserialize(tune_net_path) >> net;
        // layer<57>(net).subnet() = layer<57>(pretrained_net).subnet();
    }

    // In case we have several GPUs, we can tell the dnn_trainer to make use of them.
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    // We initialize the trainer here, as it will be used in several contexts, depending on the
    // arguments passed the the program.
    auto trainer = dnn_trainer(net, sgd(weight_decay, momentum), gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_synchronization_file(sync_file_name, std::chrono::minutes(30));

    // If the training has started and a synchronization file has already been saved to disk,
    // we can re-run this program with the --test option and a confidence threshold to see
    // how the training is going.
    if (parser.option("test"))
    {
        if (!file_exists(sync_file_name))
        {
            std::cerr << "Could not find file " << sync_file_name << '\n';
            return EXIT_FAILURE;
        }
        const double threshold = get_option(parser, "test", 0.01);
        image_window win;
        rgb_image image, resized;
        for (const auto& im : train_dataset.images)
        {
            win.clear_overlay();
            load_image(image, data_path + "/" + im.filename);
            win.set_title(im.filename);
            win.set_image(image);
            const auto tform = preprocess_image(image, resized, image_size);
            auto detections = net.process(resized, threshold);
            postprocess_detections(tform, detections);
            std::clog << "# detections: " << detections.size() << '\n';
            for (const auto& det : detections)
            {
                win.add_overlay(det.rect, string_to_color(det.label), det.label);
                std::clog << det.label << ": " << det.rect << " " << det.detection_confidence
                          << '\n';
            }
            std::cin.get();
        }
        return EXIT_SUCCESS;
    }

    dlib::pipe<std::pair<rgb_image, std::vector<yolo_rect>>> test_data(10 * batch_size / num_gpus);
    const auto test_loader = [&test_data, &test_dataset, &data_path, image_size](time_t seed)
    {
        dlib::rand rnd(time(nullptr) + seed);
        while (test_data.is_enabled())
        {
            const auto idx = rnd.get_random_64bit_number() % test_dataset.images.size();
            std::pair<rgb_image, std::vector<yolo_rect>> sample;
            rgb_image image;
            const auto& image_info = test_dataset.images.at(idx);
            try
            {
                load_image(image, data_path + "/" + image_info.filename);
            }
            catch (const image_load_error& e)
            {
                std::cerr << "ERROR: " << e.what() << std::endl;
                sample.first.set_size(image_size, image_size);
                assign_all_pixels(sample.first, rgb_pixel(0, 0, 0));
                sample.second = {};
                test_data.enqueue(sample);
                continue;
            }
            const rectangle_transform tform = letterbox_image(image, sample.first, image_size);
            for (const auto& box : image_info.boxes)
                sample.second.emplace_back(tform(box.rect), 1, box.label);
            test_data.enqueue(sample);
        }
    };

    // Create some data loaders which will load the data, and perform som data augmentation.
    dlib::pipe<std::pair<rgb_image, std::vector<yolo_rect>>> train_data(100 * batch_size);
    const auto train_loader = [&](time_t seed)
    {
        dlib::rand rnd(time(nullptr) + seed);
        const auto get_sample = [&]()
        {
            std::pair<rgb_image, std::vector<yolo_rect>> result;
            rgb_image image, blurred, letterbox, transformed(image_size, image_size);
            const auto idx = rnd.get_random_64bit_number() % train_dataset.images.size();
            const auto& image_info = train_dataset.images.at(idx);
            try
            {
                load_image(image, data_path + "/" + image_info.filename);
            }
            catch (const image_load_error& e)
            {
                std::cerr << "ERROR: " << e.what() << std::endl;
                result.first.set_size(image_size, image_size);
                assign_all_pixels(result.first, rgb_pixel(0, 0, 0));
                result.second = {};
                return result;
            }
            for (const auto& box : image_info.boxes)
                result.second.emplace_back(box.rect, 1, box.label);

            // First, letterbox the image
            rectangle_transform tform(letterbox_image(image, letterbox, image_size));
            for (auto& box : result.second)
                box.rect = tform(box.rect);

            // scale, shift and rotate
            const double scale = rnd.get_double_in_range(1 - scale_gain, 1 + scale_gain);
            const auto shift_amount = shift_frac * image_size;
            const dpoint center = dpoint(image_size / 2., image_size / 2.) +
                                  dpoint(
                                      rnd.get_double_in_range(-shift_amount, shift_amount),
                                      rnd.get_double_in_range(-shift_amount, shift_amount));
            const chip_details cd(
                centered_drect(center, image_size * scale, image_size * scale),
                {image_size, image_size},
                rnd.get_double_in_range(-angle * pi / 180, angle * pi / 180));

            extract_image_chip(letterbox, cd, result.first);
            tform = get_mapping_to_chip(cd);
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
                gaussian_blur(result.first, blurred);
                result.first = blurred;
            }
            if (rnd.get_random_double() < perspective_prob)
            {
                const drectangle r(0, 0, image_size - 1, image_size - 1);
                std::array ps{r.tl_corner(), r.tr_corner(), r.bl_corner(), r.br_corner()};
                const double amount = 0.05 * image_size;
                for (auto& corner : ps)
                {
                    corner.x() += rnd.get_double_in_range(-amount, amount);
                    corner.y() += rnd.get_double_in_range(-amount, amount);
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
                    const auto tb = std::minmax({ps[0].y(), ps[1].y(), ps[2].y(), ps[3].y()});
                    box.rect.left() = lr.first;
                    box.rect.top() = tb.first;
                    box.rect.right() = lr.second;
                    box.rect.bottom() = tb.second;
                }
            }

            if (rnd.get_random_double() < color_jitter_prob)
            {
                if (rnd.get_random_double() < 0.5)
                {
                    disturb_colors(result.first, rnd, gamma_magnitude, color_magnitude);
                }
                else
                {
                    matrix<hsi_pixel> hsi;
                    assign_image(hsi, result.first);
                    const auto color_gain = 1 + color_magnitude;
                    const auto dhue = rnd.get_double_in_range(1 / color_gain, color_gain);
                    const auto dsat = rnd.get_double_in_range(1 / color_gain, color_gain);
                    const auto dexp = rnd.get_double_in_range(1 / color_gain, color_gain);
                    for (auto& p : hsi)
                    {
                        p.h = put_in_range(0, 255, p.h * dhue);
                        p.s = put_in_range(0, 255, p.s * dsat);
                        p.i = put_in_range(0, 255, p.i * dexp);
                    }
                    assign_image(result.first, hsi);
                }
            }

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

            // Ignore or remove boxes that are not well covered by the current image
            const auto image_rect = get_rect(result.first);

            if (ignore_partial_boxes)
            {
                for (auto& box : result.second)
                {
                    const auto coverage = box.rect.intersect(image_rect).area() / box.rect.area();
                    if (coverage < min_coverage)
                        box.ignore = true;
                }
            }
            else  // remove them
            {
                const auto p = std::partition(
                    result.second.begin(),
                    result.second.end(),
                    [&image_rect, min_coverage](const yolo_rect& b) {
                        return b.rect.intersect(image_rect).area() / b.rect.area() > min_coverage;
                    });
                result.second.erase(p, result.second.end());
            }

            // Finally, for the remaining boxes, make them fit inside the image rectangle
            for (auto& box : result.second)
            {
                if (not box.ignore)
                {
                    box.rect.left() = put_in_range(0, image_size, box.rect.left());
                    box.rect.top() = put_in_range(0, image_size, box.rect.top());
                    box.rect.right() = put_in_range(0, image_size, box.rect.right());
                    box.rect.bottom() = put_in_range(0, image_size, box.rect.bottom());
                }
            }

            return result;
        };

        const auto mixup = [&rnd, get_sample]()
        {
            const auto sample1 = get_sample();
            const auto sample2 = get_sample();
            std::pair<rgb_image, std::vector<yolo_rect>> sample;
            DLIB_CASSERT(have_same_dimensions(sample1.first, sample2.first));
            sample.first.set_size(sample1.first.nr(), sample1.first.nc());
            const auto alpha = rnd.get_random_beta(8, 8);
            for (long r = 0; r < sample.first.nr(); ++r)
            {
                for (long c = 0; c < sample.first.nc(); ++c)
                {
                    sample.first(r, c).red =
                        alpha * sample1.first(r, c).red + (1 - alpha) * sample2.first(r, c).red;
                    sample.first(r, c).green = alpha * sample1.first(r, c).green +
                                               (1 - alpha) * sample2.first(r, c).green;
                    sample.first(r, c).blue =
                        alpha * sample1.first(r, c).blue + (1 - alpha) * sample2.first(r, c).blue;
                }
            }
            for (auto box : sample1.second)
            {
                box.detection_confidence = alpha;
                sample.second.push_back(std::move(box));
            }
            for (auto box : sample2.second)
            {
                box.detection_confidence = 1 - alpha;
                sample.second.push_back(std::move(box));
            }
            return sample;
        };

        while (train_data.is_enabled())
        {
            if (rnd.get_random_double() < mosaic_prob)
            {
                const long s = image_size * 0.5;
                std::pair<rgb_image, std::vector<yolo_rect>> sample;
                sample.first.set_size(image_size, image_size);
                const std::vector<std::pair<long, long>> pos{{0, 0}, {0, s}, {s, 0}, {s, s}};
                for (const auto& [x, y] : pos)
                {
                    std::pair<rgb_image, std::vector<yolo_rect>> tile;
                    if (rnd.get_random_double() < mixup_prob)
                        tile = mixup();
                    else
                        tile = get_sample();
                    const rectangle r(x, y, x + s, y + s);
                    auto si = sub_image(sample.first, r);
                    resize_image(tile.first, si);
                    for (auto& box : tile.second)
                    {
                        box.rect = translate_rect(scale_rect(box.rect, 0.5), x, y);
                        sample.second.push_back(std::move(box));
                    }
                }
                train_data.enqueue(sample);
            }
            else
            {
                if (rnd.get_random_double() < mixup_prob)
                    train_data.enqueue(mixup());
                else
                    train_data.enqueue(get_sample());
            }
        }
    };

    std::vector<std::thread> train_data_loaders;
    for (size_t i = 0; i < num_workers; ++i)
        train_data_loaders.emplace_back([train_loader, i]() { train_loader(i + 1); });

    std::vector<std::thread> test_data_loaders;
    if (test_period > 0)
    {
        for (size_t i = 0; i < 2; ++i)
            test_data_loaders.emplace_back([test_loader, i]() { test_loader(i + 1); });
    }

    // It is always a good idea to visualize the training samples.  By passing the --visualize
    // flag, we can see the training samples that will be fed to the dnn_trainer.
    if (parser.option("visualize"))
    {
        image_window win;
        win.set_title("YOLO dataset visualization");
        std::clog << "Press any key to visualize the next training sample or q to quit.\n";
        while (not win.is_closed())
        {
            std::pair<rgb_image, std::vector<yolo_rect>> sample;
            train_data.dequeue(sample);
            win.clear_overlay();
            win.set_image(sample.first);
            for (const auto& r : sample.second)
            {
                auto color = string_to_color(r.label);
                // cross-out ignored boxes and make them semi-transparent
                if (r.ignore)
                {
                    color.alpha = 128;
                    win.add_overlay(r.rect.tl_corner(), r.rect.br_corner(), color);
                    win.add_overlay(r.rect.tr_corner(), r.rect.bl_corner(), color);
                }
                win.add_overlay(r.rect, color, r.label);
            }
            unsigned long key;
            bool is_printable;
            win.get_next_keypress(key, is_printable);
            if (key == 'q' or key == base_window::KEY_ESC)
                win.close_window();
        }
    }

    std::vector<rgb_image> images;
    std::vector<std::vector<yolo_rect>> bboxes;

    // The main training loop, that we will reuse for the warmup and the rest of the training.
    const auto train = [&images, &bboxes, &train_data, &test_data, &trainer, test_period]()
    {
        static size_t train_cnt = 0;
        images.clear();
        bboxes.clear();
        std::pair<rgb_image, std::vector<yolo_rect>> sample;
        if (test_period == 0 or ++train_cnt % test_period != 0)
        {
            while (images.size() < trainer.get_mini_batch_size())
            {
                train_data.dequeue(sample);
                images.push_back(std::move(sample.first));
                bboxes.push_back(std::move(sample.second));
            }
            trainer.train_one_step(images, bboxes);
        }
        else
        {
            while (images.size() < trainer.get_mini_batch_size())
            {
                test_data.dequeue(sample);
                images.push_back(std::move(sample.first));
                bboxes.push_back(std::move(sample.second));
            }
            trainer.test_one_step(images, bboxes);
        }
    };

    const auto num_steps_per_epoch = train_dataset.images.size() / trainer.get_mini_batch_size();
    const auto warmup_steps = warmup_epochs * num_steps_per_epoch;

    // The training process can be unstable at the beginning.  For this reason, we
    // exponentially increase the learning rate during the first warmup steps.
    if (trainer.get_train_one_step_calls() < warmup_steps)
    {
        if (trainer.get_train_one_step_calls() == 0)
        {
            matrix<double> learning_rate_schedule;
            if (burnin)
                learning_rate_schedule = learning_rate * pow(linspace(1e-24, 1, warmup_steps), 4);
            else
                learning_rate_schedule = linspace(1e-99, learning_rate, warmup_steps);

            trainer.set_learning_rate_schedule(learning_rate_schedule);
            std::cout << "training started with " << warmup_epochs;
            if (burnin)
                std::cout << " burn-in ";
            else
                std::cout << " linear ";
            std::cout << "warm-up epochs (" << warmup_steps << " steps)\n";
            std::cout << trainer;
        }
        while (trainer.get_train_one_step_calls() < warmup_steps)
            train();
        trainer.get_net(force_flush_to_disk::no);
        std::cout << "warm-up finished\n";
    }

    // setup the trainer after the warm-up
    if (trainer.get_train_one_step_calls() == warmup_steps)
    {
        if (cosine_epochs > 0)
        {
            const size_t cosine_steps = (cosine_epochs - warmup_epochs) * num_steps_per_epoch;
            std::cout << "training with cosine scheduler for " << cosine_epochs - warmup_epochs
                      << " epochs (" << cosine_steps << " steps)\n";
            // clang-format off
            const matrix<double> learning_rate_schedule =
            min_learning_rate + 0.5 * (learning_rate - min_learning_rate) *
            (1 + cos(linspace(0, cosine_steps, cosine_steps) * pi / cosine_steps));
            // clang-format on
            trainer.set_learning_rate_schedule(learning_rate_schedule);
        }
        else
        {
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(min_learning_rate);
            trainer.set_learning_rate_shrink_factor(0.1);
            if (test_period > 0)
            {
                trainer.set_iterations_without_progress_threshold(
                    patience * test_period * num_steps_per_epoch);
                trainer.set_test_iterations_without_progress_threshold(
                    patience * test_dataset.images.size() / trainer.get_mini_batch_size());
            }
            else
            {
                trainer.set_iterations_without_progress_threshold(patience * num_steps_per_epoch);
                trainer.set_test_iterations_without_progress_threshold(0);
            }
        }
        std::clog << trainer << '\n';
    }
    else
    {
        // we print the trainer to stderr in case we resume the training.
        std::clog << trainer << '\n';
    }

    double best_map = 0;
    double best_wf1 = 0;
    if (file_exists(best_metrics_path))
        deserialize(best_metrics_path) >> best_map >> best_wf1;
    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate())
    {
        train();
        const auto num_steps = trainer.get_train_one_step_calls();
        if (num_steps % num_steps_per_epoch == 0)
        {
            net_infer_type inet(trainer.get_net());
            const auto epoch = num_steps / num_steps_per_epoch;
            std::cerr << "computing mean average precison for epoch " << epoch << std::endl;
            dlib::pipe<image_info> data(1000);
            test_data_loader test_loader(parser[0], test_dataset, data, image_size, num_workers);
            std::thread test_loaders([&test_loader]() { test_loader.run(); });
            const auto metrics = compute_metrics(
                inet,
                test_dataset,
                2 * batch_size / num_gpus,
                data,
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
                      << "\n\n"
                      << std::flush;

            serialize(best_metrics_path) << best_map << best_wf1;

            data.disable();
            test_loaders.join();
            inet.clean();
        }
    }

    trainer.get_net();
    std::cout << trainer << '\n';
    std::cout << "training done\n";

    train_data.disable();
    for (auto& worker : train_data_loaders)
        worker.join();

    if (test_period > 0)
    {
        test_data.disable();
        for (auto& worker : test_data_loaders)
            worker.join();
    }

    serialize(experiment_name + ".dnn") << net;
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
