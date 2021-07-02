#include "rgpnet.h"
#include "webcam_window.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/videoio.hpp>
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
    parser.add_option("thickness", "bounding box thickness (default: 5)", 1);
    parser.add_option("no-labels", "do not draw label names");
    parser.add_option("font", "path to custom bdf font", 1);
    parser.add_option("multilabel", "draw multiple labels");
    parser.add_option("nms", "IoU and area covered ratio thresholds (default: 0.45 1)", 2);
    parser.add_option("classwise-nms", "classwise NMS");
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
    parser.check_incompatible_options("multilabel", "no-labels");
    parser.check_option_arg_range<size_t>("size", 32, std::numeric_limits<size_t>::max());
    parser.check_option_arg_range<size_t>("thickness", 0, 10);
    parser.check_option_arg_range<double>("nms", 0, 1);

    const size_t image_size = dlib::get_option(parser, "size", 512);
    const std::string dnn_path = dlib::get_option(parser, "dnn", "");
    const std::string sync_path = dlib::get_option(parser, "sync", "");
    const size_t thickness = dlib::get_option(parser, "thickness", 5);
    const std::string font_path = dlib::get_option(parser, "font", "");
    const bool classwise_nms = parser.option("classwise-nms");
    double iou_threshold = 0.45;
    double ratio_covered = 1.0;
    if (parser.option("nms"))
    {
        iou_threshold = std::stod(parser.option("nms").argument(0));
        ratio_covered = std::stod(parser.option("nms").argument(1));
    }

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

    net.loss_details().adjust_nms(iou_threshold, ratio_covered, classwise_nms);
    std::cout << net.loss_details() << std::endl;

    draw_options options(font_path);
    options.thickness = thickness;
    options.multilabel = parser.option("multilabel");
    options.draw_labels = not parser.option("no-labels");
    for (const auto& label : net.loss_details().get_options().labels)
        options.string_to_color(label);

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
        draw_bounding_boxes(image, detections, options);
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
