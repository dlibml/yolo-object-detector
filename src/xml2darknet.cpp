#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace dlib;

// For Darknet compatibility
const std::map<std::string, unsigned long> coco_mapping{
    {"person", 0},         {"bicycle", 1},       {"car", 2},
    {"motorcycle", 3},     {"airplane", 4},      {"bus", 5},
    {"train", 6},          {"truck", 7},         {"boat", 8},
    {"traffic light", 9},  {"fire hydrant", 10}, {"stop sign", 11},
    {"parking meter", 12}, {"bench", 13},        {"bird", 14},
    {"cat", 15},           {"dog", 16},          {"horse", 17},
    {"sheep", 18},         {"cow", 19},          {"elephant", 20},
    {"bear", 21},          {"zebra", 22},        {"giraffe", 23},
    {"backpack", 24},      {"umbrella", 25},     {"handbag", 26},
    {"tie", 27},           {"suitcase", 28},     {"frisbee", 29},
    {"skis", 30},          {"snowboard", 31},    {"sports ball", 32},
    {"kite", 33},          {"baseball bat", 34}, {"baseball glove", 35},
    {"skateboard", 36},    {"surfboard", 37},    {"tennis racket", 38},
    {"bottle", 39},        {"wine glass", 40},   {"cup", 41},
    {"fork", 42},          {"knife", 43},        {"spoon", 44},
    {"bowl", 45},          {"banana", 46},       {"apple", 47},
    {"sandwich", 48},      {"orange", 49},       {"broccoli", 50},
    {"carrot", 51},        {"hot dog", 52},      {"pizza", 53},
    {"donut", 54},         {"cake", 55},         {"chair", 56},
    {"couch", 57},         {"potted plant", 58}, {"bed", 59},
    {"dining table", 60},  {"toilet", 61},       {"tv", 62},
    {"laptop", 63},        {"mouse", 64},        {"remote", 65},
    {"keyboard", 66},      {"cell phone", 67},   {"microwave", 68},
    {"oven", 69},          {"toaster", 70},      {"sink", 71},
    {"refrigerator", 72},  {"book", 73},         {"clock", 74},
    {"vase", 75},          {"scissors", 76},     {"teddy bear", 77},
    {"hair drier", 78},    {"toothbrush", 79},
};

auto main(const int argc, const char** argv) -> int
try
{
    const auto num_threads = std::thread::hardware_concurrency();
    const auto num_threads_str = std::to_string(num_threads);
    command_line_parser parser;
    parser.add_option("coco", "use the default coco mapping");
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


        // map each label to a class index
        std::map<std::string, unsigned long> labels_map;
        if (parser.option("coco"))
        {
            labels_map = coco_mapping;
        }
        else
        {
            // get the unique labels from the dataset
            std::set<std::string> labels_set;
            for (const auto& image : dataset.images)
            {
                for (const auto& box : image.boxes)
                {
                    labels_set.insert(box.label);
                }
            }
            // generate a mapping
            size_t i = 0;
            for (const auto& label : labels_set)
            {
                labels_map[label] = i++;
                std::cout << label << " => " << labels_map.at(label) << '\n';
            }
        }

        parallel_for_verbose(
            num_workers,
            0,
            dataset.images.size(),
            [&](size_t i)
            {
                const auto& im = dataset.images.at(i);
                const auto label_path =
                    output_path / fs::path(im.filename).replace_extension(".txt");
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
                    try
                    {
                        fout << labels_map.at(box.label) << ' ' << p.x() / width << ' '
                             << p.y() / height << ' ' << r.width() / width << ' '
                             << r.height() / height << '\n';
                    }
                    catch (const std::out_of_range& e)
                    {
                        std::cerr << e.what() << '\n';
                        std::cout << box.label << std::endl;
                    }
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
