#include "model.h"
#include "detector_utils.h"
#include "drawing_utils.h"
#include "webcam_window.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/videoio.hpp>
#include <tools/imglab/src/metadata_editor.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;
using fseconds = std::chrono::duration<float>;
using fms = std::chrono::duration<float, std::milli>;
const auto image_types = dlib::match_endings(".jpg .jpeg .gif .png .bmp .JPG JPEG .GIF .PNG .BMP");

auto main(const int argc, const char** argv) -> int
try
{
    dlib::command_line_parser parser;
    parser.set_group_name("Detector Options");
    parser.add_option("conf", "detection confidence threshold (default: 0.25)", 1);
    parser.add_option("dnn", "load this network file", 1);
    parser.add_option("nms", "IoU and area covered thresholds (default: 0.45 1)", 2);
    parser.add_option("no-classwise", "disable classwise NMS");
    parser.add_option("size", "image size for inference (default: 512)", 1);
    parser.add_option("sync", "load this sync file", 1);
    parser.set_group_name("Display Options");
    parser.add_option("fill", "fill bounding boxes with transparency", 1);
    parser.add_option("font", "path to custom bdf font", 1);
    parser.add_option("mapping", "mapping file to change labels names", 1);
    parser.add_option("multilabel", "allow multiple labels per detection");
    parser.add_option("no-conf", "do not display the confidence value");
    parser.add_option("no-labels", "do not draw label names");
    parser.add_option("thickness", "bounding box thickness (default: 5)", 1);
    parser.set_group_name("I/O Options");
    parser.add_option("fps", "force frames per second (default: 30)", 1);
    parser.add_option("input", "input file to process instead of the camera", 1);
    parser.add_option("image", "path to image file", 1);
    parser.add_option("images", "path to directory with images", 1);
    parser.add_option("output", "output file to write out the processed input", 1);
    parser.add_option("webcam", "webcam device to use (default: 0)", 1);
    parser.add_option("xml", "export the network to xml and exit", 1);
    parser.set_group_name("Pseudo-labelling Options");
    parser.add_option("check", "check that all files in the dataset exist");
    parser.add_option("update", "update this dataset with pseudo-labels", 1);
    parser.add_option("overlap", "overlap between truth and pseudo-labels", 2);
    parser.set_group_name("Help Options");
    parser.add_option("architecture", "print the network architecture and exit");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        webcam_window::print_keyboard_shortcuts();
        return EXIT_SUCCESS;
    }

    net_infer_type net;

    if (parser.option("architecture"))
    {
        std::cout << net << std::endl;
        return EXIT_SUCCESS;
    }

    // check for incompatible input options
    const auto input_options = std::array{"image", "images", "input", "webcam"};
    for (size_t i = 0; i < input_options.size(); ++i)
    {
        for (size_t j = i + 1; j < input_options.size(); ++j)
        {
            parser.check_incompatible_options(input_options[i], input_options[j]);
        }
    }

    parser.check_incompatible_options("dnn", "sync");
    parser.check_incompatible_options("no-labels", "multilabel");
    parser.check_incompatible_options("no-labels", "font");
    parser.check_incompatible_options("no-labels", "mapping");
    parser.check_option_arg_range<size_t>("size", 224, 2048);
    parser.check_option_arg_range<size_t>("fill", 0, 255);
    parser.check_option_arg_range<size_t>("thickness", 0, 10);
    parser.check_option_arg_range<double>("conf", 0, 1);
    parser.check_option_arg_range<double>("nms", 0, 1);
    parser.check_option_arg_range<double>("overlap", 0, 1);
    parser.check_sub_option("update", "overlap");
    parser.check_sub_option("update", "check");

    const size_t image_size = dlib::get_option(parser, "size", 512);
    const double conf_thresh = dlib::get_option(parser, "conf", 0.25);
    const std::string dnn_path = dlib::get_option(parser, "dnn", "");
    const std::string sync_path = dlib::get_option(parser, "sync", "");
    const std::string font_path = dlib::get_option(parser, "font", "");
    const bool classwise_nms = not parser.option("no-classwise");
    const size_t webcam_index = dlib::get_option(parser, "webcam", 0);
    const std::string input_path = dlib::get_option(parser, "input", "");
    const std::string output_path = dlib::get_option(parser, "output", "");
    const std::string mapping_path = dlib::get_option(parser, "mapping", "");
    const std::string dataset_path = dlib::get_option(parser, "update", "");
    const std::string xml_path = dlib::get_option(parser, "xml", "");
    float fps = dlib::get_option(parser, "fps", 30);
    double nms_iou_threshold = 0.45;
    double nms_ratio_covered = 1.0;
    if (parser.option("nms"))
    {
        nms_iou_threshold = std::stod(parser.option("nms").argument(0));
        nms_ratio_covered = std::stod(parser.option("nms").argument(1));
    }

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

    net.loss_details().adjust_nms(nms_iou_threshold, nms_ratio_covered, classwise_nms);
    std::cout << net.loss_details() << std::endl;

    if (not xml_path.empty())
    {
        std::clog << "exporting net to " << xml_path << std::endl;
        dlib::net_to_xml(net, xml_path);
        return EXIT_SUCCESS;
    }

    if (not dataset_path.empty())
    {
        const bool check_dataset = parser.option("check");
        dlib::image_dataset_metadata::dataset dataset;
        dlib::image_dataset_metadata::load_image_dataset_metadata(dataset, dataset_path);
        dlib::locally_change_current_dir chdir(get_parent_directory(dlib::file(dataset_path)));
        rgb_image image, letterbox;
        double overlap_iou_threshold = 0.45;
        double overlap_ratio_covered = 1;
        if (parser.option("overlap"))
        {
            overlap_iou_threshold = std::stod(parser.option("overlap").argument(0));
            overlap_ratio_covered = std::stod(parser.option("overlap").argument(1));
        }
        dlib::test_box_overlap overlaps(overlap_iou_threshold, overlap_ratio_covered);
        dlib::console_progress_indicator progress(dataset.images.size());
        for (size_t i = 0; i < dataset.images.size(); ++i)
        {
            auto& image_info = dataset.images[i];
            if (check_dataset)
            {
                if (not dlib::file_exists(image_info.filename))
                    std::cout << image_info.filename << '\n';
                continue;
            }
            dlib::load_image(image, image_info.filename);
            const auto tform = preprocess_image(image, letterbox, image_size);
            auto detections = net.process(letterbox, conf_thresh);
            postprocess_detections(tform, detections);
            std::vector<dlib::image_dataset_metadata::box> boxes;
            for (const auto& pseudo : detections)
            {
                if (not overlaps_any_box(image_info.boxes, pseudo, overlaps, classwise_nms))
                {
                    dlib::image_dataset_metadata::box box;
                    box.rect = pseudo.rect;
                    box.label = pseudo.label;
                    image_info.boxes.push_back(std::move(box));
                }
            }
            progress.print_status(i + 1, false, std::cerr);
        }
        std::cerr << std::endl;
        chdir.revert();
        const auto new_path = dataset_path.substr(0, dataset_path.rfind('.')) + "-pseudo.xml";
        dlib::image_dataset_metadata::save_image_dataset_metadata(dataset, new_path);
        return EXIT_SUCCESS;
    }

    drawing_options options(font_path);
    options.fill = dlib::get_option(parser, "fill", 0);
    options.thickness = dlib::get_option(parser, "thickness", 5);
    options.multilabel = parser.option("multilabel");
    options.draw_labels = not parser.option("no-labels");
    options.draw_confidence = not parser.option("no-conf") and options.draw_labels;
    for (const auto& label : net.loss_details().get_options().labels)
    {
        options.string_to_color(label);
        options.mapping[label] = label;
    }

    if (not mapping_path.empty())
    {
        std::ifstream fin(mapping_path);
        if (not fin.good())
            throw std::runtime_error("Error reading " + mapping_path);
        std::string line;
        for (const auto& label : net.loss_details().get_options().labels)
        {
            getline(fin, line);
            std::cerr << "mapping: " << label << " => " << line << std::endl;
            options.mapping.at(label) = line;
        }
    }

    webcam_window win(conf_thresh);
    win.can_record = not output_path.empty();

    if (parser.option("image"))
    {
        rgb_image image, letterbox;
        dlib::load_image(image, parser.option("image").argument());
        const auto tform = preprocess_image(image, letterbox, image_size);
        const auto t0 = std::chrono::steady_clock::now();
        auto detections = net.process(letterbox, conf_thresh);
        const auto t1 = std::chrono::steady_clock::now();
        const auto t = std::chrono::duration_cast<fms>(t1 - t0).count();
        std::cout << parser.option("image").argument() << ": " << t << " ms" << std::endl;
        postprocess_detections(tform, detections);
        for (const auto& d : detections)
        {
            std::cout << d.label << " " << d.detection_confidence << ": " << d.rect << "\n";
        }
        draw_bounding_boxes(image, detections, options);
        if (not output_path.empty())
            dlib::save_png(image, output_path);
        win.set_image(image);
        win.wait_until_closed();
        return EXIT_SUCCESS;
    }

    if (parser.option("images"))
    {
        rgb_image image, letterbox;
        const auto path = parser.option("images").argument();
        const auto files = dlib::get_files_in_directory_tree(path, image_types);
        std::cout << "# images: " << files.size() << std::endl;
        for (const auto& file : files)
        {
            dlib::load_image(image, file.full_name());
            const auto tform = preprocess_image(image, letterbox, image_size);
            const auto t0 = std::chrono::steady_clock::now();
            auto detections = net.process(letterbox, conf_thresh);
            const auto t1 = std::chrono::steady_clock::now();
            postprocess_detections(tform, detections);
            const auto t = std::chrono::duration_cast<fms>(t1 - t0).count();
            std::cout << file.full_name() << ": " << t << " ms" << std::endl;
            for (const auto& d : detections)
            {
                std::cout << d.label << " " << d.detection_confidence << ": " << d.rect << "\n";
            }
            draw_bounding_boxes(image, detections, options);
            if (not output_path.empty())
                dlib::save_png(image, output_path);
            win.set_image(image);
            std::cin.get();
        }
        return EXIT_SUCCESS;
    }

    cv::VideoCapture vid_src;
    cv::VideoWriter vid_snk;
    if (not input_path.empty())
    {
        cv::VideoCapture file(input_path);
        if (not parser.option("fps"))
            fps = file.get(cv::CAP_PROP_FPS);
        vid_src = file;
        win.mirror = false;
        if (not output_path.empty())
            win.recording = true;
    }
    else
    {
        cv::VideoCapture cap(webcam_index);
        cap.set(cv::CAP_PROP_FPS, fps);
        vid_src = cap;
        win.mirror = true;
    }

    int width, height;
    {
        cv::Mat cv_tmp;
        vid_src.read(cv_tmp);
        width = cv_tmp.cols;
        height = cv_tmp.rows;
    }

    if (not output_path.empty())
    {
        vid_snk = cv::VideoWriter(
            output_path,
            cv::VideoWriter::fourcc('X', '2', '6', '4'),
            fps,
            cv::Size(width, height));
    }

    rgb_image image, letterbox;
    dlib::running_stats_decayed<float> det_fps(100);
    while (not win.is_closed())
    {
        cv::Mat cv_cap;
        if (not vid_src.read(cv_cap))
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
        if (win.recording and not output_path.empty())
        {
            dlib::matrix<dlib::bgr_pixel> bgr_img(height, width);
            dlib::assign_image(bgr_img, image);
            vid_snk.write(dlib::toMat(bgr_img));
        }
    }
    if (not output_path.empty())
        vid_snk.release();
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
