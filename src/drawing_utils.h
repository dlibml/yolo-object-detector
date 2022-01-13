#ifndef drawing_utils_h_INCLUDED
#define drawing_utils_h_INCLUDED

#include <dlib/image_transforms.h>
#include <tools/imglab/src/metadata_editor.h>

struct drawing_options
{
    drawing_options() = default;
    drawing_options(const std::string& font_path)
    {
        if (not font_path.empty())
        {
            const auto font = std::make_shared<dlib::bdf_font>();
            std::ifstream fin(font_path);
            if (fin.good())
            {
                font->read_bdf_file(fin, 0xFFFF);
                font->adjust_metrics();
                custom_font = std::move(font);
            }
            else
            {
                std::cerr << "WARNING: could not open font file " + font_path +
                                 ", using default font."
                          << std::endl;
            }
        }
    };
    size_t thickness = 5;
    color_mapper string_to_color;
    bool draw_labels = true;
    bool draw_confidence = true;
    bool multilabel = false;
    uint8_t fill = 0;
    std::map<std::string, std::string> mapping;
    const std::shared_ptr<dlib::font>& get_font()
    {
        if (custom_font != nullptr)
            font = custom_font;
        else
            font = default_font;
        return font;
    }

    private:
    const std::shared_ptr<dlib::font> default_font = dlib::default_font::get_font();
    std::shared_ptr<dlib::bdf_font> custom_font;
    std::shared_ptr<dlib::font> font;
};

void draw_bounding_boxes(
    dlib::matrix<dlib::rgb_pixel>& image,
    const std::vector<dlib::yolo_rect>& detections,
    drawing_options& opts);

#endif  // drawing_utils_h_INCLUDED
