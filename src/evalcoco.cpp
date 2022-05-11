#include "detector_utils.h"
#include "drawing_utils.h"
#include "model.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <nlohmann/json.hpp>

using namespace dlib;
using json = nlohmann::json;

auto round_decimal_places(const double val, const unsigned int n = 2) -> double
{
    const auto rounder = std::pow(10., n);
    return std::round(val * rounder) / rounder;
}

auto main(const int argc, const char** argv) -> int
try
{
    command_line_parser parser;
    parser.add_option("json", "path to the image_info_test-dev2017.json", 1);
    parser.add_option("size", "image size to process images (default: 640)", 1);
    parser.add_option("dnn", "path to the network to evaluate", 1);
    parser.add_option("conf", "detection confidence threshold (default: 0.001)", 1);
    parser.add_option("letterbox", "force letter box on single inference");

    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    parser.parse(argc, argv);

    const long image_size = get_option(parser, "size", 640);
    const double conf_thresh = get_option(parser, "conf", 0.001);
    const bool use_letterbox = parser.option("letterbox");
    const std::string json_path = get_option(parser, "json", "");
    if (json_path.empty())
    {
        std::cerr << "Specify the path to image_info_test-dev2017.json with --json\n";
        return EXIT_FAILURE;
    }

    const std::string dnn_path = get_option(parser, "dnn", "");
    if (dnn_path.empty())
    {
        std::cerr << "Specify the path to the network to evaluate with --dnn\n";
        return EXIT_FAILURE;
    }

    std::ifstream fin(json_path);
    if (not fin.good())
        throw std::runtime_error("ERROR while trying to open " + json_path + " file.");

    json data;
    fin >> data;

    std::unordered_map<std::string, int> categories;
    for (const auto& entry : data["categories"])
    {
        categories[entry["name"].get<std::string>()] = entry["id"].get<int>();
    }

    model net;
    net.load_infer(dnn_path);

    image_window win;
    drawing_options options;
    options.draw_labels = true;
    for (const auto& label : net.get_options().labels)
        options.mapping[label] = label;
    matrix<rgb_pixel> image, resized;
    const auto images_path = get_parent_directory(file(json_path)).full_name() + "/test2017";
    json results;
    console_progress_indicator progress(data["images"].size());
    size_t cnt = 0;
    for (const auto& image_info : data["images"])
    {
        const auto image_id = image_info["id"].get<int>();
        load_image(image, images_path + "/" + image_info["file_name"].get<std::string>());
        const auto tform = preprocess_image(image, resized, image_size, use_letterbox);
        auto dets = net(resized, conf_thresh);
        postprocess_detections(tform, dets);
        for (const auto& det : dets)
        {
            const double x = round_decimal_places(det.rect.left(), 1);
            const double y = round_decimal_places(det.rect.top(), 1);
            const double w = round_decimal_places(det.rect.width(), 1);
            const double h = round_decimal_places(det.rect.height(), 1);
            const auto d = json{
                {"image_id", image_id},
                {"category_id", categories.at(det.label)},
                {"bbox", json{x, y, w, h}},
                {"score", round_decimal_places(det.detection_confidence, 3)}};
            results.push_back(std::move(d));
        }
        progress.print_status(++cnt, false, std::clog);

        // draw_bounding_boxes(image, dets, options);
        // win.set_image(image);
        // std::cout << results.back().dump(2) << '\n';
        // std::cin.get();
        // break;

    }
    std::clog << "saving results\n";
    std::ofstream fout("detections_test-dev2017_yolo-dlib_results.json");
    fout << results << std::flush;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
