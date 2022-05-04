#include "detector_utils.h"
#include "drawing_utils.h"
#include "model.h"
#include "sgd_trainer.h"
#include "webcam_window.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <filesystem>
#include <opencv2/videoio.hpp>
#include <tools/imglab/src/metadata_editor.h>

namespace fs = std::filesystem;
using namespace dlib;
using rgb_image = matrix<rgb_pixel>;
using fseconds = std::chrono::duration<float>;
using fms = std::chrono::duration<float, std::milli>;
const std::unordered_set<std::string> image_exts{
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
};

auto main(const int argc, const char** argv) -> int
try
{
    command_line_parser parser;
    parser.set_group_name("Detector Options");
    parser.add_option("conf", "detection confidence threshold (default: 0.25)", 1);
    parser.add_option("dnn", "load this network file", 1);
    parser.add_option("fuse", "fuse network layers and save the net", 1);
    parser.add_option("nms", "IoU and area covered thresholds (default: 0.45 1)", 2);
    parser.add_option("no-classwise", "disable classwise NMS");
    parser.add_option("size", "image long side for inference (default: 512)", 1);
    parser.add_option("sync", "load this sync file", 1);

    parser.set_group_name("Display Options");
    parser.add_option("fill", "fill bounding boxes with transparency", 1);
    parser.add_option("font", "path to custom bdf font", 1);
    parser.add_option("mapping", "mapping file to change labels names", 1);
    parser.add_option("multilabel", "allow multiple labels per detection");
    parser.add_option("no-conf", "do not display the confidence value");
    parser.add_option("no-labels", "do not draw label names");
    parser.add_option("thickness", "bounding box thickness (default: 5)", 1);
    parser.add_option("load-options", "load drawing options file", 1);
    parser.add_option("save-options", "save drawing options file", 1);
    parser.add_option("offset", "fix text position (default: 0 0)", 2);
    parser.add_option("weighted", "use confidence as thickness");

    parser.set_group_name("I/O Options");
    parser.add_option("fps", "force frames per second (default: 30)", 1);
    parser.add_option("input", "input file to process instead of the camera", 1);
    parser.add_option("image", "path to image file", 1);
    parser.add_option("images", "path to directory with images", 1);
    parser.add_option("letterbox", "force letter box on single inference");
    parser.add_option("output", "output file to write out the processed input", 1);
    parser.add_option("webcam", "webcam device to use (default: 0)", 1);

    parser.set_group_name("Pseudo-labelling Options");
    parser.add_option("dry-run", "check that all files in the dataset exist");
    parser.add_option("pseudo", "update this dataset with pseudo-labels", 1);
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

    model net;

    if (parser.option("architecture"))
    {
        net.print(std::clog);
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
    parser.check_incompatible_options("no-labels", "offset");
    parser.check_incompatible_options("no-labels", "mapping");
    parser.check_option_arg_range<size_t>("size", 224, 8192);
    parser.check_option_arg_range<size_t>("fill", 0, 255);
    parser.check_option_arg_range<size_t>("thickness", 0, 10);
    parser.check_option_arg_range<double>("conf", 0, 1);
    parser.check_option_arg_range<double>("nms", 0, 1);
    parser.check_option_arg_range<double>("overlap", 0, 1);
    parser.check_sub_option("pseudo", "overlap");
    parser.check_sub_option("pseudo", "dry-run");

    const size_t image_size = get_option(parser, "size", 512);
    const double conf_thresh = get_option(parser, "conf", 0.25);
    const std::string dnn_path = get_option(parser, "dnn", "");
    const std::string sync_path = get_option(parser, "sync", "");
    const std::string font_path = get_option(parser, "font", "");
    const bool classwise_nms = not parser.option("no-classwise");
    const size_t webcam_index = get_option(parser, "webcam", 0);
    const std::string input_path = get_option(parser, "input", "");
    fs::path output_path = get_option(parser, "output", "");
    const std::string mapping_path = get_option(parser, "mapping", "");
    const std::string dataset_path = get_option(parser, "pseudo", "");
    const std::string fused_path = get_option(parser, "fuse", "");
    const bool use_letterbox = parser.option("letterbox");
    float fps = get_option(parser, "fps", 30);
    double nms_iou_threshold = 0.45;
    double nms_ratio_covered = 1.0;
    if (parser.option("nms"))
    {
        nms_iou_threshold = std::stod(parser.option("nms").argument(0));
        nms_ratio_covered = std::stod(parser.option("nms").argument(1));
    }
    point text_offset(0, 0);
    if (parser.option("offset"))
    {
        text_offset.x() = std::stoi(parser.option("offset").argument(0));
        text_offset.y() = std::stoi(parser.option("offset").argument(1));
    }

    // Try to load the network from either a weights file or a trainer state
    if (not dnn_path.empty())
    {
        net.load_infer(dnn_path);
    }
    else if (not sync_path.empty() and file_exists(sync_path))
    {
        auto trainer = sgd_trainer(net);
        trainer.load_from_synchronization_file(sync_path);
        std::cerr << "Lodaded network from " << sync_path << std::endl;
        std::cerr << "learning rate:  " << trainer.get_learning_rate() << std::endl;
        std::cerr << "training steps: " << trainer.get_train_one_step_calls() << std::endl;
    }
    else
    {
        std::cerr << "ERROR: could not load the network.\n";
        return EXIT_FAILURE;
    }

    // General options for drawing bounding boxes on images
    drawing_options options;
    options.set_font(font_path);
    for (const auto& label : net.get_options().labels)
    {
        options.string_to_color(label);
        options.mapping[label] = label;
    }

    if (parser.option("load-options"))
    {
        deserialize(parser.option("load-options").argument()) >> options;
        if (parser.option("font"))
            options.set_font(font_path);
        if (parser.option("fill"))
            options.fill = get_option(parser, "fill", 0);
        if (parser.option("thickness"))
            options.thickness = get_option(parser, "thickness", 5);
        if (parser.option("multilabel"))
            options.multilabel = parser.option("multilabel");
        if (parser.option("no-labels"))
            options.draw_labels = not parser.option("no-labels");
        if (parser.option("no-conf"))
            options.draw_confidence = not parser.option("no-conf") and options.draw_labels;
        if (parser.option("weighted"))
            options.weighted = parser.option("weighted");
        if (parser.option("offset"))
            options.text_offset = text_offset;
    }
    else
    {
        options.fill = get_option(parser, "fill", 0);
        options.thickness = get_option(parser, "thickness", 5);
        options.multilabel = parser.option("multilabel");
        options.draw_labels = not parser.option("no-labels");
        options.draw_confidence = not parser.option("no-conf") and options.draw_labels;
        options.weighted = parser.option("weighted");
        options.text_offset = text_offset;
    }
    if (not mapping_path.empty())
    {
        std::ifstream fin(mapping_path);
        if (not fin.good())
            throw std::runtime_error("Error reading " + mapping_path);
        std::string line;
        for (const auto& label : net.get_options().labels)
        {
            getline(fin, line);
            std::cerr << "mapping: " << label << " => " << line << std::endl;
            options.mapping.at(label) = line;
        }
    }

    if (parser.option("save-options"))
        serialize(parser.option("save-options").argument()) << options;

    // Setup the loss nms
    net.adjust_nms(nms_iou_threshold, nms_ratio_covered, classwise_nms);
    net.print_loss_details();

    // Fuse layers
    if (not fused_path.empty())
    {
        std::clog << "fusing layers and saving net to " << fused_path << '\n';
        net.fuse();
        net.save_infer(fused_path);
    }

    // Process the dataset if for pseudo labeling
    if (not dataset_path.empty())
    {
        const bool check_dataset = parser.option("dry-run");
        image_dataset_metadata::dataset dataset;
        image_dataset_metadata::load_image_dataset_metadata(dataset, dataset_path);
        locally_change_current_dir chdir(get_parent_directory(file(dataset_path)));
        rgb_image image, resized;
        double overlap_iou_threshold = 0.45;
        double overlap_ratio_covered = 1;
        if (parser.option("overlap"))
        {
            overlap_iou_threshold = std::stod(parser.option("overlap").argument(0));
            overlap_ratio_covered = std::stod(parser.option("overlap").argument(1));
        }
        test_box_overlap overlaps(overlap_iou_threshold, overlap_ratio_covered);
        console_progress_indicator progress(dataset.images.size());
        for (size_t i = 0; i < dataset.images.size(); ++i)
        {
            auto& image_info = dataset.images[i];
            if (check_dataset)
            {
                if (not fs::exists(image_info.filename))
                    std::clog << image_info.filename << '\n';
                continue;
            }
            load_image(image, image_info.filename);
            image_info.width = image.nc();
            image_info.height = image.nr();
            const auto tform = preprocess_image(image, resized, image_size, use_letterbox);
            auto detections = net(resized, conf_thresh);
            postprocess_detections(tform, detections);
            for (const auto& pseudo : detections)
            {
                if (not overlaps_any_box(image_info.boxes, pseudo, overlaps, classwise_nms))
                {
                    image_dataset_metadata::box box(pseudo.rect);
                    box.label = pseudo.label;
                    image_info.boxes.push_back(std::move(box));
                }
            }
            progress.print_status(i + 1, false, std::cerr);
        }
        std::cerr << std::endl;
        chdir.revert();
        const auto new_path = dataset_path.substr(0, dataset_path.rfind('.')) + "-pseudo.xml";
        image_dataset_metadata::save_image_dataset_metadata(dataset, new_path);
        return EXIT_SUCCESS;
    }

    webcam_window win(options, conf_thresh);
    win.can_record = not output_path.empty();

    if (parser.option("image"))
    {
        rgb_image image, letterbox;
        load_image(image, parser.option("image").argument());
        const auto tform = preprocess_image(image, letterbox, image_size, use_letterbox);
        const auto t0 = std::chrono::steady_clock::now();
        auto detections = net(letterbox, win.conf_thresh);
        const auto t1 = std::chrono::steady_clock::now();
        const auto t = std::chrono::duration_cast<fms>(t1 - t0).count();
        std::clog << parser.option("image").argument() << ": " << t << " ms" << std::endl;
        postprocess_detections(tform, detections);
        for (const auto& d : detections)
        {
            std::clog << d.label << " " << d.detection_confidence << ": ";
            std::clog << center(d.rect) << " " << d.rect.width() << "x" << d.rect.height();
            std::clog << "\n";
        }
        std::clog << "Total number of detections: " << detections.size() << std::endl;
        draw_bounding_boxes(image, detections, options);
        if (not output_path.empty())
        {
            const auto ext = output_path.extension();
            if (ext == ".jpg")
                save_jpeg(image, output_path);
            else if (ext == ".png")
                save_png(image, output_path);
            else if (ext == ".bmp")
                save_bmp(image, output_path);
            else if (ext == ".dng")
                save_dng(image, output_path);
            else if (ext == ".webp")
                save_webp(image, output_path);
            else  // save to lossless WebP otherwise (unknown or empty extension)
                save_webp(image, output_path.replace_extension(".webp"), 101);
        }
        win.set_title(parser.option("image").argument());
        win.set_image(image);
        win.wait_until_closed();
        return EXIT_SUCCESS;
    }

    if (parser.option("images"))
    {
        rgb_image image, letterbox;
        std::vector<fs::path> files;
        const fs::path path = parser.option("images").argument();
        fs::create_directories(output_path / path.relative_path());
        for (const auto& item : fs::recursive_directory_iterator(path))
        {
            if (item.is_directory())
            {
                std::cout << output_path / item.path() << std::endl;
                fs::create_directories(output_path / item.path());
            }
            else if (
                item.is_regular_file() and
                image_exts.count(tolower(item.path().extension().string())))
            {
                files.push_back(item.path());
            }
        }
        std::clog << "# images: " << files.size() << '\n';
        console_progress_indicator progress(files.size());
        for (size_t i = 0; i < files.size(); ++i)
        {
            auto& file = files[i];
            load_image(image, file);
            const auto tform = preprocess_image(image, letterbox, image_size, use_letterbox);
            const auto t0 = std::chrono::steady_clock::now();
            auto detections = net(letterbox, win.conf_thresh);
            const auto t1 = std::chrono::steady_clock::now();
            postprocess_detections(tform, detections);
            const auto t = std::chrono::duration_cast<fms>(t1 - t0).count();
            draw_bounding_boxes(image, detections, options);
            if (output_path.empty())
            {
                std::clog << file << ": " << t << " ms\n";
                for (const auto& d : detections)
                {
                    std::clog << d.label << " " << d.detection_confidence << ": ";
                    std::clog << center(d.rect) << " " << d.rect.width() << "x" << d.rect.height();
                    std::clog << "\n";
                }
                std::clog << "Total number of detections: " << detections.size() << '\n';
                win.set_title(file.filename());
                win.set_image(image);
                std::cin.get();
            }
            else
            {
                save_webp(image, output_path / file.replace_extension(".webp"), 101);
                progress.print_status(i + 1, false, std::cerr);
            }
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
        std::clog << "original image size: " << width << 'x' << height << '\n';
    }

    if (not output_path.empty())
    {
        vid_snk = cv::VideoWriter(
            output_path,
            cv::VideoWriter::fourcc('X', '2', '6', '4'),
            fps,
            cv::Size(width, height));
    }

    rgb_image image, resized;
    running_stats_decayed<float> det_fps(100);
    while (not win.is_closed())
    {
        cv::Mat cv_cap;
        if (not vid_src.read(cv_cap))
            break;
        const cv_image<bgr_pixel> tmp(cv_cap);
        if (win.mirror)
            flip_image_left_right(tmp, image);
        else
            assign_image(image, tmp);

        const auto t0 = std::chrono::steady_clock::now();
        const auto tform = preprocess_image(image, resized, image_size, use_letterbox);
        auto detections = net(resized, win.conf_thresh);
        postprocess_detections(tform, detections);
        const auto t1 = std::chrono::steady_clock::now();
        draw_bounding_boxes(image, detections, options);
        win.set_image(image);
        det_fps.add(1.0f / std::chrono::duration_cast<fseconds>(t1 - t0).count());
        std::clog << "size: " << resized.nc() << 'x' << resized.nr() << ", fps: " << det_fps.mean()
                  << "              \r" << std::flush;
        if (win.recording and not output_path.empty())
        {
            matrix<bgr_pixel> bgr_img(height, width);
            assign_image(bgr_img, image);
            vid_snk.write(toMat(bgr_img));
        }
    }
    if (not output_path.empty())
        vid_snk.release();
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
