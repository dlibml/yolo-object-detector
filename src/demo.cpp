#include "rgpnet.h"
#include "webcam_window.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>
#include <tools/imglab/src/metadata_editor.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;
using fseconds = std::chrono::duration<float>;
using fms = std::chrono::duration<float, std::milli>;

dlib::rectangle_transform
    preprocess_image(const rgb_image& image, rgb_image& output, const long image_size)
{
    return dlib::rectangle_transform(inv(letterbox_image(image, output, image_size)));
}

void postprocess_detections(
    const dlib::rectangle_transform& tform,
    std::vector<dlib::yolo_rect>& detections)
{
    for (auto& d : detections)
        d.rect = tform(d.rect);
}

auto main(const int argc, const char** argv) -> int
try
{
    dlib::command_line_parser parser;
    parser.add_option("dnn", "load this network file", 1);
    parser.add_option("sync", "load this sync file", 1);
    parser.add_option("size", "image size for inference (default: 512)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }
    parser.check_incompatible_options("dnn", "sync");

    const size_t image_size = dlib::get_option(parser, "size", 512);
    const std::string dnn_path = dlib::get_option(parser, "dnn", "");
    const std::string sync_path = dlib::get_option(parser, "sync", "");

    rgpnet::infer net;

    if (not dnn_path.empty())
    {
        dlib::deserialize(dnn_path) >> net;
    }
    else if (not sync_path.empty() and dlib::file_exists(sync_path))
    {
        auto trainer = dlib::dnn_trainer(net);
        trainer.set_synchronization_file(sync_path);
        trainer.get_net();
    }
    else
    {
        std::cout << "ERROR: could not load the network." << std::endl;
        return EXIT_FAILURE;
    }

    color_mapper string_to_color;
    for (const auto& label : net.loss_details().get_options().labels)
    {
        std::cout << label << std::endl;
        string_to_color(label);
    }

    webcam_window win;
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FPS, 30);
    rgb_image image, letterbox;
    dlib::running_stats_decayed<float> det_fps(100);
    while (not win.is_closed())
    {
        cv::Mat cv_cap;
        if (not cap.read(cv_cap))
            break;
        const dlib::cv_image<dlib::bgr_pixel> tmp(cv_cap);
        if (win.mirror)
            dlib::flip_image_left_right(tmp, image);
        else
            dlib::assign_image(image, tmp);

        const auto t0 = std::chrono::steady_clock::now();
        const auto tform = preprocess_image(image, letterbox, image_size);
        auto detections = net.process(letterbox, win.conf_thresh);
        postprocess_detections(tform, detections);
        const auto t1 = std::chrono::steady_clock::now();
        render_bounding_boxes(image, detections, string_to_color);
        win.set_image(image);
        det_fps.add(1.0f / std::chrono::duration_cast<fseconds>(t1 - t0).count());
        std::cout << "FPS: " << det_fps.mean() << "              \r" << std::flush;
    }
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
