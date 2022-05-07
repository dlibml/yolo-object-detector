#include "model.h"
#include "sgd_trainer.h"

#include <dlib/cmd_line_parser.h>

using namespace dlib;
using fms = std::chrono::duration<float, std::milli>;

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

    const std::string output_path = get_option(parser, "output", "fused.dnn");

    model net;
    std::clog << "loading network from " << parser[0];
    auto t0 = std::chrono::steady_clock::now();
    net.load_infer(parser[0]);
    auto t1 = std::chrono::steady_clock::now();
    std::clog << " (" << std::chrono::duration_cast<fms>(t1 - t0).count() << " ms)\n";
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
