#include "detector_utils.h"
#include "metrics.h"
#include "model.h"
#include "sgd_trainer.h"

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
    parser.add_option("name", "name used for net and sync files (default: yolo)", 1);
    parser.add_option("size", "image size for internal usage (default: 512)", 1);
    parser.add_option("test", "visually test the model instead of training");
    parser.add_option("visualize", "visualize data augmentation before training");
    parser.add_option("conf", "threshold used for testing (default 0.25)", 1);

    parser.set_group_name("Training Options");
    parser.add_option("backbone", "use this pre-trained backbone", 1);
    parser.add_option("batch-gpu", "mini batch size per GPU (default: 8)", 1);
    parser.add_option("gpus", "number of GPUs for the training (default: 1)", 1);
    parser.add_option("tune", "path to the network to fine-tune", 1);
    parser.add_option("workers", "number data loaders (default: " + num_threads_str + ")", 1);

    parser.set_group_name("Scheduler Options");
    parser.add_option("burnin", "use exponential burn-in (default: 1.0)", 1);
    parser.add_option("epochs", "total epochs for linear scheduler (default: 0.0)", 1);
    parser.add_option("cosine", "use cosine scheduler instead of linear");
    parser.add_option("learning-rate", "initial learning rate (default: 0.001)", 1);
    parser.add_option("min-learning-rate", "minimum learning rate (default: 1e-6)", 1);
    parser.add_option("shrink-factor", "learning rate shrink factor (default: 0.1)", 1);
    parser.add_option("patience", "number of epochs without progress (default: 3.0)", 1);
    parser.add_option("test-period", "test a batch every <arg> steps (default: 0)", 1);
    parser.add_option("warmup", "number warm-up epochs (default: 0.0)", 1);

    parser.set_group_name("Optimizer Options");
    parser.add_option("momentum", "sgd momentum (default: 0.9)", 1);
    parser.add_option("weight-decay", "sgd weight decay (default: 0.0005)", 1);

    parser.set_group_name("YOLO Options");
    parser.add_option("box", "anchor box pyramid level, width and height", 3);
    parser.add_option("iou-ignore", "IoUs above don't incur obj loss (default: 0.7)", 1);
    parser.add_option("iou-anchor", "extra anchors IoU threshold (default: 0.2)", 1);
    parser.add_option("lambda-obj", "weight for the objectness loss (default: 1)", 1);
    parser.add_option("lambda-box", "weight for the box regression loss (default: 1)", 1);
    parser.add_option("lambda-cls", "weight for the classification loss (default: 1)", 1);
    parser.add_option("beta-cls", "class balanced loss beta (default: disabled)", 1);
    parser.add_option("gamma-obj", "focal loss gamma for the objectness (default: 0)", 1);
    parser.add_option("gamma-cls", "focal loss gamma for the classifier (default: 0)", 1);

    parser.set_group_name("Data Augmentation Options");
    parser.add_option("angle", "max rotation in degrees (default: 3.0)", 1);
    parser.add_option("hsi", "HSI colorspace gains (default: 0.5 0.2 0.1)", 3);
    parser.add_option("ignore-partial", "ignore partially covered objects instead");
    parser.add_option("min-coverage", "remove partially covered objects (default: 0.5)", 1);
    parser.add_option("mirror", "mirror probability (default: 0.5)", 1);
    parser.add_option("mixup", "mixup probability (default: 0.0)", 1);
    parser.add_option("mosaic", "mosaic probability (default: 0.5)", 1);
    parser.add_option("perspective", "relative to image size (default: 0.01)", 1);
    parser.add_option("scale", "scale gain (default: 0.5)", 1);
    parser.add_option("shift", "shift relative to image size (default: 0.2)", 1);
    parser.add_option("solarize", "probability of solarize (default: 0.0)", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");

    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/DIRECTORY\n";
        parser.print_options();
        std::cout << "Give the path to a directory with the training.xml and testing.xml files.\n";
        return EXIT_SUCCESS;
    }
    const auto epsilon = std::numeric_limits<double>::epsilon();
    parser.check_option_arg_range<double>("conf", 0, 1);
    parser.check_option_arg_range<double>("iou-ignore", 0, 1);
    parser.check_option_arg_range<double>("iou-anchor", 0, 1);
    parser.check_option_arg_range<double>("gamma-obj", 0, std::numeric_limits<double>::max());
    parser.check_option_arg_range<double>("gamma-cls", 0, std::numeric_limits<double>::max());
    parser.check_option_arg_range<double>("beta-cls", 0, 1 - epsilon);
    parser.check_option_arg_range<double>("mirror", 0, 1);
    parser.check_option_arg_range<double>("mixup", 0, 1);
    parser.check_option_arg_range<double>("mosaic", 0, 1);
    parser.check_option_arg_range<double>("scale", 0, 1);
    parser.check_option_arg_range<double>("perspective", 0, 1);
    parser.check_option_arg_range<double>("min-coverage", 0, 1);
    parser.check_option_arg_range<double>("hsi", 0, 1);
    parser.check_option_arg_range<double>("shrink-factor", 1e-99, 1);
    parser.check_incompatible_options("epochs", "patience");
    parser.check_incompatible_options("epochs", "shrink-factor");
    parser.check_incompatible_options("backbone", "tune");
    parser.check_sub_option("epochs", "cosine");
    parser.check_sub_option("warmup", "burnin");
    const double learning_rate = get_option(parser, "learning-rate", 0.001);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-6);
    const double patience = get_option(parser, "patience", 3.0);
    const double shrink_factor = get_option(parser, "shrink-factor", 0.1);
    const double warmup_epochs = get_option(parser, "warmup", 0.0);
    const double num_epochs = get_option(parser, "epochs", 0.0);
    if (parser.option("epochs"))
        DLIB_CASSERT(num_epochs > warmup_epochs);
    double gain_h = 0.5, gain_s = 0.2, gain_i = 0.1;
    if (parser.option("hsi"))
    {
        gain_h = std::stod(parser.option("hsi").argument(0));
        gain_s = std::stod(parser.option("hsi").argument(1));
        gain_i = std::stod(parser.option("hsi").argument(2));
    }
    const double test_conf = get_option(parser, "conf", 0.25);
    const double lambda_obj = get_option(parser, "lambda-obj", 1.0);
    const double lambda_box = get_option(parser, "lambda-box", 1.0);
    const double lambda_cls = get_option(parser, "lambda-cls", 1.0);
    const double gamma_obj = get_option(parser, "gamma-obj", 0.0);
    const double gamma_cls = get_option(parser, "gamma-cls", 0.0);
    const double beta_cls = get_option(parser, "beta-cls", 0.0);
    const size_t num_gpus = get_option(parser, "gpus", 1);
    const size_t batch_size = get_option(parser, "batch-gpu", 8) * num_gpus;
    const double burnin = get_option(parser, "burnin", 1.0);
    const size_t test_period = get_option(parser, "test-period", 0);
    const size_t image_size = get_option(parser, "size", 512);
    const size_t num_workers = get_option(parser, "workers", num_threads);
    const double mirror_prob = get_option(parser, "mirror", 0.5);
    const double mosaic_prob = get_option(parser, "mosaic", 0.5);
    const double mixup_prob = get_option(parser, "mixup", 0.0);
    const double perspective_frac = get_option(parser, "perspective", 0.01);
    const double angle = get_option(parser, "angle", 3);
    const double scale_gain = get_option(parser, "scale", 0.5);
    const double shift_frac = get_option(parser, "shift", 0.2);
    const double min_coverage = get_option(parser, "min-coverage", 0.5);
    const bool ignore_partial_boxes = parser.option("ignore-partial");
    const double solarize_prob = get_option(parser, "solarize", 0.0);
    const double iou_ignore_threshold = get_option(parser, "iou-ignore", 0.7);
    const double iou_anchor_threshold = get_option(parser, "iou-anchor", 0.2);
    const float momentum = get_option(parser, "momentum", 0.9);
    const float weight_decay = get_option(parser, "weight-decay", 0.0005);
    const std::string experiment_name = get_option(parser, "name", "yolo");
    const std::string sync_file_name = experiment_name + "_sync";
    const std::string net_file_name = experiment_name + ".dnn";
    const std::string best_metrics_path = experiment_name + "_best_metrics.dat";
    const std::string backbone_path = get_option(parser, "backbone", "");
    const std::string tune_net_path = get_option(parser, "tune", "");

    // Path to the data directory containing training.xml and testing.xml
    const std::string data_path = parser[0];
    image_dataset_metadata::dataset train_dataset;
    image_dataset_metadata::load_image_dataset_metadata(
        train_dataset,
        data_path + "/training.xml");
    std::clog << "# train images: " << train_dataset.images.size() << '\n';
    std::map<std::string, size_t> class_support;
    std::map<std::string, double> class_weights;
    size_t num_objects = 0;
    // compute the label support
    for (const auto& im : train_dataset.images)
    {
        for (const auto& b : im.boxes)
        {
            class_support[b.label]++;
            ++num_objects;
        }
    }
    // compute the class balanced loss weights
    if (not parser.option("beta-cls"))
    {
        for (const auto& [label, support] : class_support)
        {
            class_weights[label] = 1;
        }
    }
    else
    {
        double sum_inv_weights = 0;
        // double sum_weights = 0;
        for (const auto& [label, support] : class_support)
        {
            class_weights[label] = (1.0 - beta_cls) / (1.0 - std::pow(beta_cls, support));
            sum_inv_weights += 1.0 / class_weights.at(label);
            // sum_weights += class_weights.at(label);
        }
        for (auto& [label, weight] : class_weights)
        {
            const auto eff_support = 1.0 / weight / sum_inv_weights * num_objects;
            weight = (num_objects - eff_support) / eff_support;
            // weight = weight / sum_weights * class_weights.size();
        }
    }

    std::clog << "# labels: " << class_support.size() << '\n';
    image_dataset_metadata::dataset test_dataset;
    image_dataset_metadata::load_image_dataset_metadata(test_dataset, data_path + "/testing.xml");
    std::clog << "# test images: " << test_dataset.images.size() << '\n';

    // YOLO options
    yolo_options options;
    color_mapper string_to_color;
    for (const auto& [label, support] : class_support)
    {
        std::clog << " - " << label << ": " << support;
        std::clog << " (" << (100.0 * support) / num_objects << "%),";
        std::clog << " weight: " << class_weights.at(label) << '\n';
        options.labels.push_back(label);
        string_to_color(label);
    }
    options.iou_ignore_threshold = iou_ignore_threshold;
    options.iou_anchor_threshold = iou_anchor_threshold;
    options.lambda_obj = lambda_obj;
    options.lambda_box = lambda_box;
    options.lambda_cls = lambda_cls;
    options.gamma_obj = gamma_obj;
    options.gamma_cls = gamma_cls;

    // Initialize the default YOLO anchors
    std::map<unsigned long, std::vector<yolo_options::anchor_box_details>> anchors{
        {3, {{10, 13}, {16, 30}, {33, 23}}},
        {4, {{30, 61}, {62, 45}, {59, 119}}},
        {5, {{116, 90}, {156, 198}, {373, 326}}}};
    if (parser.option("box"))
    {
        anchors.clear();
        const auto num_anchors = parser.option("anchor").count();
        for (size_t i = 0; i < num_anchors; ++i)
        {
            const auto stride = std::stoul(parser.option("anchor").argument(0, i));
            const auto width = std::stoul(parser.option("anchor").argument(1, i));
            const auto height = std::stoul(parser.option("anchor").argument(2, i));
            anchors[stride].emplace_back(width, height);
        }
        for (auto& [stride, anchor] : anchors)
            std::sort(
                anchor.begin(),
                anchor.end(),
                [](const auto& a, const auto& b)
                { return a.width * a.height < b.width * b.height; });
    }

    // Add the anchors to the YOLO options
    try
    {
        options.add_anchors<ytag3>(anchors.at(3));
        options.add_anchors<ytag4>(anchors.at(4));
        options.add_anchors<ytag5>(anchors.at(5));
    }
    catch (const std::out_of_range&)
    {
        throw std::length_error("ERROR: wrong or missing pyramid level specified in anchor.");
    }

    model net(options);
    if (parser.option("architecture"))
    {
        rgb_image dummy(image_size, image_size);
        net(dummy);
        net.print(std::clog);
    }

    if (not backbone_path.empty())
    {
        net.load_backbone(backbone_path);
    }
    if (not tune_net_path.empty())
    {
        net.load_train(tune_net_path);
    }

    net.setup(options);

    // In case we have several GPUs, we can tell the dnn_trainer to make use of them.
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    // We initialize the trainer here, as it will be used in several contexts, depending on the
    // arguments passed the the program.
    auto trainer = sgd_trainer(net, weight_decay, momentum, gpus);
    trainer.be_verbose();
    trainer.set_mini_batch_size(batch_size);
    trainer.set_synchronization_file(sync_file_name);

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
        image_window win;
        rgb_image image, resized;
        net.sync();
        for (const auto& im : train_dataset.images)
        {
            win.clear_overlay();
            load_image(image, data_path + "/" + im.filename);
            win.set_title(im.filename);
            win.set_image(image);
            const auto tform = preprocess_image(image, resized, image_size);
            auto detections = net(resized, test_conf);
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

    // Create some data loaders which will load the data, and perform some data augmentation.
    dlib::pipe<std::pair<rgb_image, std::vector<yolo_rect>>> train_data(100 * batch_size);
    const auto train_loader = [&](time_t seed)
    {
        dlib::rand rnd(time(nullptr) + seed);
        const auto get_sample = [&]()
        {
            std::pair<rgb_image, std::vector<yolo_rect>> result;
            rgb_image image, letterbox, transformed(image_size, image_size);
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
                result.second.emplace_back(box.rect, class_weights.at(box.label), box.label);

            // First, letterbox the image
            rectangle_transform tform(letterbox_image(image, letterbox, image_size));
            for (auto& box : result.second)
                box.rect = tform(box.rect);

            // Scale, shift and rotate
            const double scale = rnd.get_double_in_range(1 - scale_gain, 1 + scale_gain);
            const auto shift = shift_frac * image_size;
            const dpoint center = dpoint(image_size / 2., image_size / 2.) +
                                  dpoint(
                                      rnd.get_double_in_range(-shift, shift),
                                      rnd.get_double_in_range(-shift, shift));
            const chip_details chip(
                centered_drect(center, image_size * scale, image_size * scale),
                {image_size, image_size},
                rnd.get_double_in_range(-angle * pi / 180, angle * pi / 180));

            extract_image_chip(letterbox, chip, result.first);
            tform = get_mapping_to_chip(chip);
            for (auto& box : result.second)
                box.rect = tform(box.rect);

            // Mirroring
            if (rnd.get_random_double() < mirror_prob)
            {
                tform = flip_image_left_right(result.first);
                for (auto& box : result.second)
                    box.rect = tform(box.rect);
            }

            // Perspective
            if (perspective_frac > 0)
            {
                const drectangle r = get_rect(result.first);
                std::array ps{r.tl_corner(), r.tr_corner(), r.bl_corner(), r.br_corner()};
                const double perspective_amount = perspective_frac * image_size;
                for (auto& corner : ps)
                {
                    corner.x() += rnd.get_double_in_range(-perspective_amount, perspective_amount);
                    corner.y() += rnd.get_double_in_range(-perspective_amount, perspective_amount);
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

            if (gain_h > 0 or gain_s > 0 or gain_i > 0)
            {
                matrix<hsi_pixel> hsi;
                assign_image(hsi, result.first);
                const auto dhue = rnd.get_double_in_range(1 / (1 + gain_h), (1 + gain_h));
                const auto dsat = rnd.get_double_in_range(1 / (1 + gain_s), (1 + gain_s));
                const auto dexp = rnd.get_double_in_range(1 / (1 + gain_i), (1 + gain_i));
                for (auto& p : hsi)
                {
                    p.h = put_in_range(0, 255, p.h * dhue);
                    p.s = put_in_range(0, 255, p.s * dsat);
                    p.i = put_in_range(0, 255, p.i * dexp);
                }
                assign_image(result.first, hsi);
                disturb_colors(result.first, rnd);
                apply_random_color_offset(result.first, rnd);
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
                        return b.rect.intersect(image_rect).area() / b.rect.area() >= min_coverage;
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

        const auto mixup = [&rnd, &class_weights, get_sample]()
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
                box.detection_confidence = alpha * class_weights.at(box.label);
                sample.second.push_back(std::move(box));
            }
            for (auto box : sample2.second)
            {
                box.detection_confidence = (1 - alpha) * class_weights.at(box.label);
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
    // gradually increase the learning rate during the first warmup steps.
    if (trainer.get_train_one_step_calls() < warmup_steps)
    {
        if (trainer.get_train_one_step_calls() == 0)
        {
            const matrix<double> learning_rate_schedule =
                learning_rate * pow(linspace(epsilon, 1, warmup_steps), burnin);

            trainer.set_learning_rate_schedule(learning_rate_schedule);
            std::cout << "training started with " << warmup_epochs;
            if (burnin > 1.0)
                std::cout << " burn-in ";
            else
                std::cout << " linear ";
            std::cout << "warm-up epochs (" << warmup_steps << " steps)\n";
            std::cout << "image size: " << image_size << '\n';
            trainer.print(std::cout);
        }
        while (trainer.get_train_one_step_calls() < warmup_steps)
            train();
        trainer.get_net(force_flush_to_disk::no);
        std::cout << "warm-up finished\n";
    }

    // setup the trainer after the warm-up
    if (trainer.get_train_one_step_calls() == warmup_steps)
    {
        if (num_epochs > 0)
        {
            const size_t num_training_steps = (num_epochs - warmup_epochs) * num_steps_per_epoch;
            matrix<double> learning_rate_schedule;
            if (parser.option("cosine"))
            {
                std::cout << "training with cosine scheduler for " << num_epochs - warmup_epochs
                          << " epochs (" << num_training_steps << " steps)\n";
                learning_rate_schedule =
                    min_learning_rate +
                    0.5 * (learning_rate - min_learning_rate) *
                        (1 + cos(linspace(0, num_training_steps, num_training_steps) * pi /
                                 num_training_steps));
            }
            else
            {
                std::cout << "training with linear scheduler for " << num_epochs - warmup_epochs
                          << " epochs (" << num_training_steps << " steps)\n";
                learning_rate_schedule =
                    linspace(learning_rate, min_learning_rate, num_training_steps);
            }
            trainer.set_learning_rate_schedule(learning_rate_schedule);
        }
        else
        {
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(min_learning_rate);
            trainer.set_learning_rate_shrink_factor(shrink_factor);
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
        trainer.get_net(force_flush_to_disk::yes);
        trainer.print(std::cout);
    }
    else
    {
        // we print the trainer to stderr in case we resume the training.
        trainer.print(std::clog);
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
            trainer.get_net();
            net.sync();
            const auto epoch = num_steps / num_steps_per_epoch;
            std::cerr << "computing mean average precison for epoch " << epoch << std::endl;
            dlib::pipe<image_info> data(1000);
            test_data_loader test_loader(parser[0], test_dataset, data, image_size, num_workers);
            std::thread test_loaders([&test_loader]() { test_loader.run(); });
            const auto metrics = compute_metrics(
                net,
                test_dataset,
                2 * batch_size / num_gpus,
                data,
                test_conf,
                std::clog);

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
            net.clean();
        }
    }

    trainer.get_net();
    trainer.print(std::cout);
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

    net.save_train(experiment_name + ".dnn");
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
