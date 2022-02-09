#include "metrics.h"
#include "model.h"
#include "sgd_trainer.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>

using namespace dlib;
using rgb_image = matrix<rgb_pixel>;

auto main(const int argc, const char** argv) -> int
try
{
    const auto num_threads = std::thread::hardware_concurrency();
    command_line_parser parser;
    parser.add_option("batch", "batch size for inference (default: 32)", 1);
    parser.add_option("conf", "detection confidence threshold (default: 0.25)", 1);
    parser.add_option("dnn", "load this network file", 1);
    parser.add_option("nms", "IoU and area covered ratio thresholds (default: 0.45 1)", 2);
    parser.add_option("nms-agnostic", "class-agnositc NMS");
    parser.add_option("size", "image size for inference (default: 512)", 1);
    parser.add_option("sync", "load this sync file", 1);
    parser.add_option(
        "workers",
        "number of data loaders (default: " + std::to_string(num_threads) + ")",
        1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("architecture", "print the network architecture");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/FILE.xml\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }
    parser.check_incompatible_options("dnn", "sync");
    parser.check_option_arg_range<size_t>("size", 224, 2048);
    parser.check_option_arg_range<double>("nms", 0, 1);

    const size_t batch_size = get_option(parser, "batch", 32);
    const size_t image_size = get_option(parser, "size", 512);
    const size_t num_workers = get_option(parser, "workers", num_threads);
    const double conf_thresh = get_option(parser, "conf", 0.25);
    const std::string dnn_path = get_option(parser, "dnn", "");
    const std::string sync_path = get_option(parser, "sync", "");
    const bool classwise_nms = not parser.option("nms-agnostic");
    double iou_threshold = 0.45;
    double ratio_covered = 1.0;
    if (parser.option("nms"))
    {
        iou_threshold = std::stod(parser.option("nms").argument(0));
        ratio_covered = std::stod(parser.option("nms").argument(1));
    }

    bool export_model = false;
    size_t num_steps = 0;

    model_infer net;
    model_train net_train;
    if (not dnn_path.empty())
    {
        net.load(dnn_path);
    }
    else if (not sync_path.empty() and file_exists(sync_path))
    {
        auto trainer = sgd_trainer(net_train);
        trainer.set_synchronization_file(sync_path);
        net = trainer.get_net();
        num_steps = trainer.get_train_one_step_calls();
        std::clog << "Lodaded network from " << sync_path << '\n';
        std::clog << "learning rate:  " << trainer.get_learning_rate() << '\n';
        std::clog << "training steps: " << num_steps << '\n';
        export_model = true;
    }
    else
    {
        std::cerr << "ERROR: could not load the network.\n";
        return EXIT_FAILURE;
    }

    net.loss_details().adjust_nms(iou_threshold, ratio_covered, classwise_nms);
    if (parser.option("architecture"))
        std::clog << net << '\n';

    net.print_loss_details();

    const auto dataset_dir = get_parent_directory(file(parser[0])).full_name();
    image_dataset_metadata::dataset dataset;
    image_dataset_metadata::load_image_dataset_metadata(dataset, parser[0]);
    dlib::pipe<image_info> data(1000);
    test_data_loader data_loader(dataset_dir, dataset, data, image_size, num_workers);

    // start the data loaders
    std::thread data_loaders([&data_loader]() { data_loader.run(); });

    const auto metrics = compute_metrics(net, dataset, batch_size, data, conf_thresh);

    data.disable();
    data_loaders.join();

    if (export_model)
        save_model(net_train, sync_path, num_steps, metrics.map, metrics.weighted_f);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
