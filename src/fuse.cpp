#include "model.h"
#include "sgd_trainer.h"
#include <filesystem>

#include <dlib/cmd_line_parser.h>

using namespace dlib;
using fms = std::chrono::duration<float, std::milli>;
namespace fs = std::filesystem;

auto main(const int argc, const char** argv) -> int
try
{
    command_line_parser parser;
    parser.add_option("output", "path to the fused network (default: fused.dnn", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }
    const fs::path net_path(parser[0]);

    const fs::path output_path = get_option(parser, "output", "fused.dnn");

    model net;
    std::clog << "loading network from " << net_path;
    auto t0 = std::chrono::steady_clock::now();
    if (net_path.extension() == ".dnn")
    {
        net.load_infer(net_path);
    }
    else
    {
        sgd_trainer trainer(net);
        trainer.load_from_synchronization_file(net_path);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::clog << " (" << std::chrono::duration_cast<fms>(t1 - t0).count() << " ms)\n";
    const auto strides = net.get_strides();
    std::cout << "the network has " << strides.size() << " outputs with strides:\n";
    for (const auto stride : strides)
        std::cout << " - " << stride << '\n';
    net.print_loss_details();
    std::clog << "fusing layers";
    t0 = std::chrono::steady_clock::now();
    net.fuse();
    t1 = std::chrono::steady_clock::now();
    std::clog << " (" << std::chrono::duration_cast<fms>(t1 - t0).count() << " ms)\n";
    std::clog << "saving network to " << output_path;
    t0 = std::chrono::steady_clock::now();
    net.save_infer(output_path);
    t1 = std::chrono::steady_clock::now();
    std::clog << " (" << std::chrono::duration_cast<fms>(t1 - t0).count() << " ms)\n";

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
