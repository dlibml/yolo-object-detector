#ifndef drawing_utils_h_INCLUDED
#define drawing_utils_h_INCLUDED

#include <dlib/image_transforms.h>
#include <tools/imglab/src/metadata_editor.h>

struct drawing_options
{
    drawing_options() = default;
    size_t thickness = 5;
    color_mapper string_to_color;
    bool draw_labels = true;
    bool draw_confidence = true;
    bool multilabel = false;
    uint8_t fill = 0;
    bool weighted = false;
    dlib::point text_offset{0, 0};
    std::map<std::string, std::string> mapping;

    auto operator=(const drawing_options& item) -> drawing_options&;
    auto get_font() -> const std::shared_ptr<dlib::font>&;
    auto set_font(const std::string& font_path) -> void;

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
