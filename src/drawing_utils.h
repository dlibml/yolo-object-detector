#ifndef drawing_utils_h_INCLUDED
#define drawing_utils_h_INCLUDED

#include <dlib/image_transforms.h>
#include <tools/imglab/src/metadata_editor.h>

struct drawing_options
{
    drawing_options() = default;
    drawing_options& operator=(const drawing_options& item)
    {
        if (this == &item)
            return *this;
        thickness = item.thickness;
        draw_labels = item.draw_labels;
        draw_confidence = item.draw_confidence;
        multilabel = item.multilabel;
        fill = item.fill;
        mapping = item.mapping;
        font_path = item.font_path;
        weighted = item.weighted;
        return *this;
    }
    size_t thickness = 5;
    color_mapper string_to_color;
    bool draw_labels = true;
    bool draw_confidence = true;
    bool multilabel = false;
    uint8_t fill = 0;
    bool weighted = false;
    std::map<std::string, std::string> mapping;
    const std::shared_ptr<dlib::font>& get_font()
    {
        if (custom_font != nullptr)
            font = custom_font;
        else
            font = default_font;
        return font;
    }
    void set_font(const std::string& font_path)
    {
        this->font_path = font_path;
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
    }

    private:
    const std::shared_ptr<dlib::font> default_font = dlib::default_font::get_font();
    std::shared_ptr<dlib::bdf_font> custom_font;
    std::shared_ptr<dlib::font> font;
    std::string font_path{};

    friend void serialize(const drawing_options& item, std::ostream& out);
    friend void deserialize(drawing_options& item, std::istream& in);
};

void draw_bounding_boxes(
    dlib::matrix<dlib::rgb_pixel>& image,
    const std::vector<dlib::yolo_rect>& detections,
    drawing_options& opts);

#endif  // drawing_utils_h_INCLUDED
