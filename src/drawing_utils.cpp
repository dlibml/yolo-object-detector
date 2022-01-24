#include "drawing_utils.h"

using namespace dlib;

void serialize(const drawing_options& item, std::ostream& out)
{
    serialize("drawing_options", out);
    serialize(item.font_path, out);
    serialize(item.thickness, out);
    serialize(item.draw_labels, out);
    serialize(item.draw_confidence, out);
    serialize(item.multilabel, out);
    serialize(item.fill, out);
    serialize(item.weighted, out);
    serialize(item.mapping, out);
    serialize(item.text_offset, out);
}
void deserialize(drawing_options& item, std::istream& in)
{
    std::string version;
    deserialize(version, in);
    if (version != "drawing_options")
        throw serialization_error("error while deserializing drawing_options");
    deserialize(item.font_path, in);
    item.set_font(item.font_path);
    deserialize(item.thickness, in);
    deserialize(item.draw_labels, in);
    deserialize(item.draw_confidence, in);
    deserialize(item.multilabel, in);
    deserialize(item.fill, in);
    deserialize(item.weighted, in);
    deserialize(item.mapping, in);
    deserialize(item.text_offset, in);
}

void draw_bounding_boxes(
    matrix<rgb_pixel>& image,
    const std::vector<yolo_rect>& detections,
    drawing_options& opts)
{
    // We want to draw most confident detections on top, so we iterate in reverse order
    for (auto det = detections.rbegin(); det != detections.rend(); ++det)
    {
        const auto& d = *det;
        auto offset = opts.thickness / 2;
        const auto color = opts.string_to_color(d.label);
        lab_pixel lab;
        assign_pixel(lab, color);
        rgb_pixel font_color(0, 0, 0);
        if (lab.l < 128)
            font_color = rgb_pixel(255, 255, 255);
        rectangle r(d.rect);
        r.left() = put_in_range(offset, image.nc() - 1 - offset, r.left());
        r.top() = put_in_range(offset, image.nr() - 1 - offset, r.top());
        r.right() = put_in_range(offset, image.nc() - 1 - offset, r.right());
        r.bottom() = put_in_range(offset, image.nr() - 1 - offset, r.bottom());
        if (opts.fill > 0)
        {
            const rgb_alpha_pixel fill(color.red, color.green, color.blue, opts.fill);
            fill_rect(image, r, fill);
        }
        if (opts.weighted)
        {
            offset *= d.detection_confidence;
            draw_rectangle(image, r, color, opts.thickness * d.detection_confidence);
        }
        else
        {
            draw_rectangle(image, r, color, opts.thickness);
        }

        if (opts.draw_labels)
        {
            std::ostringstream sout;
            sout << std::fixed << std::setprecision(0);
            if (opts.multilabel)
            {
                for (size_t i = 0; i < d.labels.size() - 1; ++i)
                {
                    sout << opts.mapping[d.labels[i].second];
                    if (opts.draw_confidence)
                        sout << " (" << d.labels[i].first * 100 << "%)";
                    sout << ", ";
                }
                sout << opts.mapping[d.labels[d.labels.size() - 1].second];
                if (opts.draw_confidence)
                    sout << " (" << d.labels[d.labels.size() - 1].first * 100 << "%)";
            }
            else
            {
                sout << opts.mapping[d.label];
                if (opts.draw_confidence)
                    sout << " (" << d.detection_confidence * 100 << "%)";
            }

            const ustring label = convert_utf8_to_utf32(sout.str());
            const auto [lw, lh] = compute_string_dims(label, opts.get_font());

            // the default case: label outside the top left corner of the box
            point label_pos(r.left(), r.top() - lh - offset + 1);
            label_pos += opts.text_offset;
            long bg_offset = opts.thickness;
            if (opts.weighted)
                bg_offset = bg_offset * d.detection_confidence + 1;
            rectangle bg(lw + bg_offset, lh);

            // draw label inside the bounding box (move it downwards)
            if (label_pos.y() < 0)
                label_pos += point(offset, lh + offset - 1);

            bg = move_rect(
                bg,
                label_pos.x() - offset - opts.text_offset.x(),
                label_pos.y() - opts.text_offset.y());
            if (opts.thickness == 1)
                fill_rect(image, bg, rgb_alpha_pixel(color.red, color.green, color.blue, 224));
            else
                fill_rect(image, bg, rgb_alpha_pixel(color));
            draw_rectangle(image, bg, color, 1);
            draw_string(image, label_pos, label, font_color, opts.get_font());
        }
    }
}
