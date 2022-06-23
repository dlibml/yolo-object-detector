#include "draw.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>
#include <filesystem>

using namespace dlib;

namespace fs = std::filesystem;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("fill", "fill bounding boxes with transparency", 1);
    parser.add_option("font", "path to custom bdf font", 1);
    parser.add_option("offset", "fix text position (default: 0 0)", 2);
    parser.add_option("output", "path to output directory (default: output)", 1);
    parser.add_option("no-labels", "do not draw label names");
    parser.add_option("thickness", "bounding box thickness (default: 5)", 1);
    parser.add_option("quality", "image quality factor for JPEG and WebP (default: 75)", 1);
    parser.add_option("jpeg", "save images as JPEG instead of PNG");
    parser.add_option("webp", "save images as WebP instead of PNG");
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]… PATH/TO/DATASET/FILE.xml" << std::endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }
    parser.check_incompatible_options("jpeg", "webp");
    parser.check_option_arg_range<int>("fill", 0, 255);
    parser.check_option_arg_range<float>("quality", 0, std::numeric_limits<float>::max());
    const fs::path output_path = get_option(parser, "output", "output");
    const fs::path font_path = get_option(parser, "font", "");
    const float quality = get_option(parser, "quality", 75);
    point text_offset(0, 0);
    if (parser.option("offset"))
    {
        text_offset.x() = std::stol(parser.option("offset").argument(0));
        text_offset.y() = std::stol(parser.option("offset").argument(1));
    }
    fs::create_directories(output_path);
    fs::path dataset_file = parser[0];
    const auto dataset_dir = dataset_file.parent_path();
    image_dataset_metadata::dataset dataset;
    image_dataset_metadata::load_image_dataset_metadata(dataset, dataset_file);
    drawing_options options;
    std::set<std::string> labels;
    size_t num_boxes = 0;
    for (const auto& image : dataset.images)
    {
        for (const auto& box : image.boxes)
        {
            labels.insert(box.label);
            ++num_boxes;
        }
    }
    std::clog << "Number of images: " << dataset.images.size() << "\n";
    std::clog << "Number of labels: " << labels.size() << "\n";
    std::clog << "Number of bboxes: " << num_boxes << "\n";
    for (const auto& label : labels)
    {
        options.string_to_color(label);
        options.mapping[label] = label;
    }
    options.set_font(font_path);
    options.draw_labels = not parser.option("no-labels");
    options.draw_confidence = false;
    options.thickness = get_option(parser, "thickness", 5);
    options.text_offset = text_offset;
    options.fill = get_option(parser, "fill", 0);
    for (const auto& image_info : dataset.images)
    {
        std::vector<yolo_rect> boxes;
        matrix<rgb_pixel> image;
        fs::path image_path = dataset_dir / image_info.filename;
        // file image_file(dataset_dir.full_name() + "/" + image_info.filename);
        load_image(image, image_path);
        for (const auto& box : image_info.boxes)
        {
            options.string_to_color(box.label);
            options.mapping[box.label] = box.label;
            boxes.emplace_back(box.rect, 1.0, box.label);
        }
        draw_bounding_boxes(image, boxes, options);
        if (parser.option("jpeg"))
            save_jpeg(
                image,
                output_path / image_path.filename().replace_extension(".jpg"),
                std::min<int>(quality, 100));
        else if (parser.option("webp"))
            save_webp(
                image,
                output_path / image_path.filename().replace_extension(".webp"),
                quality);
        else
            save_png(image, output_path / image_path.filename().replace_extension(".png"));
        std::cout << "press enter to save the next image.\n";
        std::cin.get();
    }
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
