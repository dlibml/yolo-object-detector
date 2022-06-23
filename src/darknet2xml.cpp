#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <dlib/image_io.h>
#include <filesystem>
#include <regex>

using namespace dlib;
namespace fs = std::filesystem;

const auto image_types =
    match_endings(".jpg .jpeg .gif .png .bmp .webp .JPG .JPEG .GIF .PNG .BMP .WEBP");

auto main(const int argc, const char** argv) -> int
try
{
    command_line_parser parser;
    auto num_threads = std::thread::hardware_concurrency();
    const auto num_threads_str = std::to_string(num_threads);
    parser.add_option("threads", "number of workers (default: " + num_threads_str + ")", 1);
    parser.add_option("names", "path to the label names file", 1);
    parser.add_option("listing", "path to the images listing file", 1);
    parser.add_option("output", "output dataset file (default: dataset.xml)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/FILE.xml" << std::endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }

    num_threads = get_option(parser, "threads", num_threads);
    const fs::path names_path = get_option(parser, "names", "");
    const fs::path listing_path = get_option(parser, "listing", "");
    if (names_path.empty())
        throw std::runtime_error("provide a names file");
    if (listing_path.empty())
        throw std::runtime_error("provide a listing file");

    // read the label names
    std::vector<std::string> names;
    {
        std::ifstream fin(names_path);
        if (not fin.good())
            throw std::runtime_error("error while opening " + std::string(argv[1]));

        for (std::string line; std::getline(fin, line);)
        {
            names.push_back(line);
        }
        std::cout << "Found " << names.size() << " names:\n";
        for (size_t i = 0; i < names.size(); ++i)
        {
            std::cout << std::fixed << std::setw(3) << i << " => " << names[i] << '\n';
        }
    }

    // read the bounding box information
    std::vector<std::string> listing;
    {
        std::ifstream fin(listing_path);
        if (not fin.good())
        {
            throw std::runtime_error("error while opening " + listing_path.string());
        }
        for (std::string line; std::getline(fin, line);)
        {
            listing.push_back(line);
        }
        std::cout << "Found " << listing.size() << " images\n";
    }

    // listings usually contain relative paths to images, so let's change directories.
    const auto dataset_dir = get_parent_directory(file(listing_path));
    locally_change_current_dir chdir(dataset_dir);

    // parse and convert the darknet labels into dlib xml format
    const std::regex image_regex{"images"};
    image_dataset_metadata::dataset dataset;
    std::mutex mutex;
    parallel_for_verbose(
        num_threads,
        0,
        listing.size(),
        [&](size_t i)
        {
            const auto& line = listing[i];
            matrix<rgb_pixel> image;
            load_image(image, line);
            image_dataset_metadata::image image_info;
            image_info.filename = line.substr(2);
            image_info.width = image.nc();
            image_info.height = image.nr();
            const auto labels_path = fs::path(std::regex_replace(line, image_regex, "labels"))
                                         .replace_extension(".txt");
            std::ifstream fl(labels_path);
            if (not fl.good())
                throw std::runtime_error("error opening " + labels_path.string());
            for (std::string label; std::getline(fl, label);)
            {
                int l;
                float x, y, w, h;
                std::istringstream sin(label);
                sin >> l >> x >> y >> w >> h;
                image_dataset_metadata::box box;
                box.label = names.at(l);
                box.rect = centered_rect(
                    std::round(x * image_info.width),
                    std::round(y * image_info.height),
                    std::round(w * image_info.width),
                    std::round(h * image_info.height));
                image_info.boxes.push_back(std::move(box));
            }
            {
                const std::lock_guard<std::mutex> lock(mutex);
                dataset.images.push_back(std::move(image_info));
            }
        });
    image_dataset_metadata::save_image_dataset_metadata(dataset, "dataset.xml");
    chdir.revert();
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
