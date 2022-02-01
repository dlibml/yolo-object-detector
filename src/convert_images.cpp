#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;

auto create_directories(const std::string& path) -> void
{
    std::istringstream sin(path);
    std::string part;
    std::vector<std::string> parts = dlib::split(path, "/");
    for (size_t k = 0; k < parts.size(); ++k)
    {
        part += parts[k] + '/';
        dlib::create_directory(part);
    }
}

auto main(const int argc, const char** argv) -> int
try
{
    const auto num_threads = std::thread::hardware_concurrency();
    dlib::command_line_parser parser;
    parser.add_option("output", "path to the output directory", 1);
    parser.add_option("quality", "JPEG quality factor (default: 75)", 1);
    parser.add_option(
        "workers",
        "number of threads (default: " + std::to_string(num_threads) + ")",
        1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    parser.check_option_arg_range<int>("quality", 0, 100);
    const int quality = get_option(parser, "quality", 75);
    const size_t num_workers = get_option(parser, "workers", num_threads);
    const std::string output_root = get_option(parser, "output", "");
    if (output_root.empty())
    {
        std::cerr << "specify the output path with --output\n";
        return EXIT_FAILURE;
    }

    create_directories(output_root);

    std::vector<dlib::file> dataset_files;
    for (size_t i = 0; i < parser.number_of_arguments(); ++i)
    {
        dataset_files.emplace_back(parser[i]);
    }

    for (const auto& dataset_file : dataset_files)
    {
        const auto input_dir = dlib::get_parent_directory(dataset_file).full_name();
        dlib::image_dataset_metadata::dataset dataset;
        load_image_dataset_metadata(dataset, dataset_file.full_name());

        std::cout << dataset_file.full_name() << '\n';
        if (not dataset.name.empty())
            std::cout << dataset.name;
        if (not dataset.comment.empty())
            std::cout << " (" << dataset.comment << ")";
        std::cout << ": " << dataset.images.size() << " images\n";

        dlib::parallel_for_verbose(
            num_workers,
            0,
            dataset.images.size(),
            [&](size_t i)
            {
                rgb_image image;
                auto& info = dataset.images[i];
                dlib::load_image(image, input_dir + "/" + info.filename);
                DLIB_CASSERT(info.width == image.nc() and info.height == image.nr());
                const auto image_path = info.filename.substr(0, info.filename.rfind('/'));
                auto image_name = dlib::right_substr(info.filename, "/");
                image_name = image_name.substr(0, image_name.rfind('.')) + ".jpg";
                const auto output_path = output_root + "/" + image_path;
                create_directories(output_root + "/" + image_path);
                dlib::save_jpeg(image, output_path + "/" + image_name, quality);
                info.filename = info.filename.substr(0, info.filename.rfind('.')) + ".jpg";
            });
        dlib::image_dataset_metadata::save_image_dataset_metadata(
            dataset,
            output_root + "/" + dataset_file.name());
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
