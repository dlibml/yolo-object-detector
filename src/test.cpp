#include "detector_utils.h"
#include "metrics.h"
#include "model.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;

auto main(const int argc, const char** argv) -> int
try
{
    const auto num_threads = std::thread::hardware_concurrency();
    dlib::command_line_parser parser;
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

    const size_t batch_size = dlib::get_option(parser, "batch", 32);
    const size_t image_size = dlib::get_option(parser, "size", 512);
    const size_t num_workers = dlib::get_option(parser, "workers", num_threads);
    const double conf_thresh = dlib::get_option(parser, "conf", 0.25);
    const std::string dnn_path = dlib::get_option(parser, "dnn", "");
    const std::string sync_path = dlib::get_option(parser, "sync", "");
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
    net_train_type net;

    if (not dnn_path.empty())
    {
        dlib::deserialize(dnn_path) >> net;
    }
    else if (not sync_path.empty() and dlib::file_exists(sync_path))
    {
        auto trainer = dlib::dnn_trainer(net);
        trainer.set_synchronization_file(sync_path);
        trainer.get_net();
        num_steps = trainer.get_train_one_step_calls();
        std::cerr << "Lodaded network from " << sync_path << std::endl;
        std::cerr << "learning rate:  " << trainer.get_learning_rate() << std::endl;
        std::cerr << "training steps: " << num_steps << std::endl;
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

    print_loss_details(net);

    dlib::pipe<image_info> data(1000);
    test_data_loader data_loader(parser[0], image_size, data, num_workers);

    // start the data loaders
    std::thread data_loaders([&data_loader]() { data_loader.run(); });

    net_infer_type tnet(net);
    const auto metrics =
        compute_metrics(tnet, data_loader.get_dataset(), batch_size, data, conf_thresh);

    data.disable();
    data_loaders.join();

    if (export_model)
        save_model(net, sync_path, num_steps, metrics.map, metrics.weighted_f);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
