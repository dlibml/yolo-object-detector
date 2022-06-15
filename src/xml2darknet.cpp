#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace dlib;

auto main(const int argc, const char** argv) -> int
try
{
    const auto num_threads = std::thread::hardware_concurrency();
    const auto num_threads_str = std::to_string(num_threads);
    command_line_parser parser;
    parser.add_option("output", "output path (default: labels)", 1);
    parser.add_option("workers", "number of worker threads (default: " + num_threads_str + ")", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/FILE.xml…\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }
    const fs::path output_path(get_option(parser, "output", "labels"));
    const size_t num_workers(get_option(parser, "workers", num_threads));

    for (size_t i = 0; i < parser.number_of_arguments(); ++i)
    {
        image_dataset_metadata::dataset dataset;
        image_dataset_metadata::load_image_dataset_metadata(dataset, parser[i]);
        fs::create_directories(output_path);

        // get the unique labels from the dataset
        std::set<std::string> labels_set;
        for (const auto& image : dataset.images)
        {
            for (const auto& box : image.boxes)
            {
                labels_set.insert(box.label);
            }
        }

        // map each label to a class index
        std::map<std::string, unsigned long> labels_map;
        {
            size_t i = 0;
            for (const auto& label : labels_set)
            {
                labels_map[label] = i++;
                std::cout << label << " => " << labels_map.at(label) << '\n';
            }
        }

        parallel_for_verbose(num_workers, 0, dataset.images.size(), [&](size_t i)
        {
            const auto& im = dataset.images.at(i);
            const auto label_path = output_path / fs::path(im.filename).replace_extension(".txt");
            if (not fs::exists(label_path.parent_path()))
                fs::create_directories(label_path.parent_path());
            std::ofstream fout(label_path);
            const double width = im.width;
            const double height = im.height;
            for (const auto& box : im.boxes)
            {
                const auto r = drectangle(
                    put_in_range(0, im.width - 1, box.rect.left()),
                    put_in_range(0, im.height - 1, box.rect.top()),
                    put_in_range(0, im.width - 1, box.rect.right()),
                    put_in_range(0, im.height - 1, box.rect.bottom()));
                const auto p = dcenter(box.rect);
                fout << labels_map.at(box.label) << ' ' << p.x() / width << ' ' << p.y() / height
                     << ' ' << r.width() / width << ' ' << r.height() / height << '\n';
            }
        });
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
