#include <dlib/cmd_line_parser.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace dlib;

const std::array<const char*, 5> supported{".jpg", ".jpeg", ".png", ".gif", ".webp"};

auto get_files(const std::string& path, const std::string& out_root) -> std::vector<std::string>
{
    std::vector<std::string> files;
    if (fs::exists("files.dat"))
    {
        std::cout << "found files.dat, deserializing\n";
        deserialize("files.dat") >> files;
    }
    else
    {
        for (const auto& item : fs::recursive_directory_iterator(path))
        {
            if (item.is_directory())
            {
                std::cout << "\r" << item << "\t\t\t\t" << std::endl;
                auto out_dir(out_root);
                fs::create_directories(out_dir.append(item.path().relative_path().string()));
            }
            else if (item.is_regular_file())
            {
                if (std::find(
                        supported.begin(),
                        supported.end(),
                        dlib::tolower(item.path().extension().string())) != supported.end())
                {
                    files.push_back(item.path().native());
                    std::cout << "scanned files: " << files.size() << "\r" << std::flush;
                }
            }
        }
        serialize("files.dat") << files;
    }
    return files;
}

auto main(const int argc, const char** argv) -> int
try
{
    command_line_parser parser;
    auto num_threads = std::thread::hardware_concurrency();
    const auto num_threads_str = std::to_string(num_threads);
    const long webp_max_dimension = 16383;
    parser.add_option("output", "output directory (default: converted_images)", 1);
    parser.add_option("threads", "number of workers (default: " + num_threads_str + ")", 1);
    parser.add_option("overwrite", "overwrite existing files");
    parser.add_option("quality", "image quality factor (default: 75.0)", 1);
    parser.add_option("log", "error log file (default: error.log)", 1);
    parser.add_option("max-side", "maximum image side (default: 16383)", 1);
    parser.add_option("min-side", "maximum image side (default: 0)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    parser.check_option_arg_range("max-side", 0, 16383);
    parser.check_option_arg_range("min-side", 0, 16383);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTIONS] DIR_1 [DIR_2...]\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const fs::path out_root(get_option(parser, "output", "converted_images"));
    auto out_path(out_root);
    out_path /= parser[0];
    if (not fs::exists(out_path))
        if (not fs::create_directories(out_path))
            throw std::runtime_error(
                "error while creating output directory: " + out_path.string());

    num_threads = get_option(parser, "threads", num_threads);
    const bool overwrite = parser.option("overwrite");
    const float quality = get_option(parser, "quality", 75.f);
    const std::string error_log = get_option(parser, "log", "error.log");
    const long max_side = get_option(parser, "max-side", webp_max_dimension);
    const long min_side = get_option(parser, "min-side", 0);

    const auto files = get_files(parser[0], out_root);

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
                const auto scale = max_side / std::max<double>(image.nr(), image.nc());
                if (scale < 1)
                    resize_image(scale, image);
                if (image.nr() < min_side or image.nc() < min_side)
                    throw std::length_error(
                        "image is too small: " + std::to_string(image.nc()) + "x" +
                        std::to_string(image.nr()));
            }
            catch (const image_load_error& e)
            {
                error = true;
                const std::lock_guard<std::mutex> lock(mutex);
                fout << file.native() << ": " << e.what() << '\n';
            }
            catch (const std::length_error& e)
            {
                error = true;
                const std::lock_guard<std::mutex> lock(mutex);
                fout << file.native() << ": " << e.what() << '\n';
            }
            if (not error)
            {
                fs::path out_file(out_root);
                out_file /= file;
                if (out_file.extension() == ".webp")
                {
                    if (not fs::copy_file(file, out_file))
                    {
                        const std::lock_guard<std::mutex> lock(mutex);
                        fout << file.native() << ": error copying file\n";
                    }
                }
                else
                {
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
            }
        });
}

catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
