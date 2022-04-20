#include <dlib/cmd_line_parser.h>
#include <dlib/image_io.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace dlib;

auto main(const int argc, const char** argv) -> int
try
{
    command_line_parser parser;
    auto num_threads = std::thread::hardware_concurrency();
    const auto num_threads_str = std::to_string(num_threads);
    parser.add_option("output", "output directory (default: converted_images)", 1);
    parser.add_option("threads", "number of workers (default: " + num_threads_str + ")", 1);
    parser.add_option("overwrite", "overwrite existing files");
    parser.add_option("quality", "image quality factor (default: 75.0)", 1);
    parser.add_option("log", "error log file (default: error.log", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTIONS] DIR_1 [DIR_2...]\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const fs::path out_root(get_option(parser, "output", "converted_images"));
    if (not fs::create_directories(out_root))
        throw std::runtime_error("error while creating output directory: " + out_root.string());

    num_threads = get_option(parser, "threads", num_threads);
    const bool overwrite = parser.option("overwrite");
    const float quality = get_option(parser, "quality", 75.f);
    const std::string error_log = get_option(parser, "log", "error.log");

    std::vector<std::string> files;
    if (fs::exists("files.dat"))
    {
        std::cout << "found files.dat, deserializing\n";
        deserialize("files.dat") >> files;
    }
    else
    {
        for (const auto& item : fs::recursive_directory_iterator(parser[0]))
        {
            if (item.is_directory())
            {
                std::cout << "\r" << item << "\t\t\t\t" << std::endl;
                auto out_dir(out_root);
                fs::create_directories(out_dir.append(item.path().relative_path().string()));
            }
            else if (item.is_regular_file())
            {
                files.push_back(item.path().native());
            }
            std::cout << "scanned files: " << files.size() << "\r" << std::flush;
        }
        serialize("files.dat") << files;
    }

    std::cout << "converting " << files.size() << " images\n";
    std::ofstream fout("error.log");
    if (not fout.good())
        throw std::runtime_error("error creating " + error_log + " file.");
    std::mutex mutex;

    parallel_for_verbose(
        num_threads,
        0,
        files.size(),
        [&](size_t i)
        {
            const fs::path file(files.at(i));
            matrix<rgb_pixel> image;
            bool error = false;
            try
            {
                load_image(image, file);
            }
            catch (const image_load_error& e)
            {
                error = true;
                const std::lock_guard<std::mutex> lock(mutex);
                fout << file.native() << ": " << e.what() << '\n';
            }
            if (not error)
            {
                fs::path out_file(out_root);
                out_file /= file;
                out_file.replace_extension(".webp");
                if (not fs::exists(out_file) or overwrite)
                {
                    try
                    {
                        save_webp(image, out_file, quality);
                    }
                    catch (const image_save_error& e)
                    {
                        const std::lock_guard<std::mutex> lock(mutex);
                        fout << file.native() << ": " << e.what() << '\n';
                    }
                }
            }
        });
}

catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
