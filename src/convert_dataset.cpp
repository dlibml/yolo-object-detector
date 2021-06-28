#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>

auto main(const int argc, const char** argv) -> int
try
{
    dlib::command_line_parser parser;
    parser.add_option("dataset", "path to the dataset XML file", 1);
    parser.add_option("workers", "number of threads (default: 8)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const std::string dataset_path = dlib::get_option(parser, "dataset", "");
    const size_t num_workers = dlib::get_option(parser, "workers", 8);
    if (dataset_path.empty())
    {
        std::cout << "specify the data path directory" << std::endl;
        return EXIT_FAILURE;
    }
    dlib::file dataset_file(dataset_path);
    const auto dataset_dir = dlib::get_parent_directory(dataset_file).full_name();

    dlib::image_dataset_metadata::dataset dataset;
    dlib::image_dataset_metadata::load_image_dataset_metadata(dataset, dataset_path);
    dlib::parallel_for_verbose(
        num_workers,
        0,
        dataset.images.size(),
        [&](size_t i = 0)
        {
            dlib::matrix<dlib::rgb_pixel> image;
            dlib::load_image(image, dataset_dir + "/" + dataset.images[i].filename);
            dataset.images[i].width = image.nc();
            dataset.images[i].height = image.nr();
        });

    dlib::image_dataset_metadata::save_image_dataset_metadata(dataset, "dataset.xml");
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
