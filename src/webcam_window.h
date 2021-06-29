#ifndef webcam_window_h_INCLUDED
#define webcam_window_h_INCLUDED

#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tools/imglab/src/metadata_editor.h>

inline auto get_random_color(const std::string& label) -> dlib::rgb_pixel
{
    int seed = 0;
    for (size_t i = 0; i < label.size(); ++i)
    {
        seed += (i + 1) * static_cast<int>(label[i]);
    }
    dlib::rand rnd(seed);
    dlib::rgb_pixel rgb(
        rnd.get_integer_in_range(0, 255),
        rnd.get_integer_in_range(0, 255),
        rnd.get_integer_in_range(0, 255));
    return rgb;
}

inline auto get_color_map(const std::vector<std::string>& labels) -> std::map<std::string, dlib::rgb_pixel>
{
    std::map<std::string, dlib::rgb_pixel> label_to_color;
    for (const auto& label : labels)
    {
        label_to_color[label] = get_random_color(label);
    }
    return label_to_color;
}

inline void render_bounding_boxes(
    dlib::matrix<dlib::rgb_pixel>& img,
    const std::vector<dlib::yolo_rect>& detections,
    color_mapper& string_to_color,
    const bool draw_labels = true)
{
    const double font_scale = 0.5;
    auto cv_img = dlib::toMat(img);
    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const auto black = cv::Scalar(0, 0, 0);
    const auto white = cv::Scalar(255, 255, 255);
    for (const auto& d : detections)
    {
        const double prob = d.detection_confidence;
        dlib::rectangle r(d.rect);
        std::ostringstream sout;
        sout << d.label << std::fixed << std::setprecision(0) << " (" << 100 * prob << "%)";
        std::string label = sout.str();
        const auto rgb = string_to_color(d.label);
        int baseline = 0;
        auto ts = cv::getTextSize(label, font, font_scale, 2, &baseline);
        const auto bbox = cv::Rect(r.left(), r.top(), r.width(), r.height());
        const auto color = cv::Scalar(rgb.red, rgb.green, rgb.blue);
        auto tr = cv::Rect(
            bbox.tl() - cv::Point(1, 0),
            cv::Point(bbox.x + ts.width + 15, bbox.y - ts.height - 20));
        cv::Point text_pos;
        cv::Rect hline, vline;
        bool draw_label = true and draw_labels;
        bool below = false;
        if (tr.y > 0)
        {
            text_pos = cv::Point(tr.x + 10, tr.y + 20);
            if (tr.width < bbox.width)
            {
                hline = cv::Rect(
                    bbox.tl() + cv::Point(-1, -3),
                    bbox.tl() + cv::Point(tr.width - 1, 2));
            }
            else
            {
                hline = cv::Rect(
                    bbox.tl() + cv::Point(-1, -3),
                    bbox.tl() + cv::Point(bbox.width + 1, 2));
            }
        }
        else
        {
            below = true;
            tr = cv::Rect(bbox.tl(), cv::Point(bbox.x + ts.width + 15, bbox.y + ts.height + 15));
            text_pos = cv::Point(tr.x + 10, tr.y + 17);
            // write only the confidence if the label is wider than the bbox
            if (static_cast<unsigned long>(tr.width) >= r.width())
            {
                sout.str("");
                sout << 100 * prob << '%';
                label = sout.str();
                ts = cv::getTextSize(label, font, font_scale, 2, &baseline);
                tr = cv::Rect(
                    bbox.tl(),
                    cv::Point(bbox.x + ts.width + 15, bbox.y + ts.height + 15));
                // if even the confidence is too wide, do not ouput the label, just color
                if (static_cast<unsigned long>(tr.width) >= r.width())
                {
                    draw_label = false;
                }
            }
            hline = cv::Rect(bbox.tl() + cv::Point(-1, 0), bbox.tl() + cv::Point(tr.width, 3));
            vline = cv::Rect(bbox.tl() + cv::Point(-1, 0), bbox.tl() + cv::Point(3, tr.height));
        }
        if (draw_label)
        {
            cv::rectangle(cv_img, tr, black, 2, cv::LINE_8, 0);
            cv::rectangle(cv_img, tr, color, cv::FILLED, cv::LINE_8, 0);
            cv::putText(cv_img, label, text_pos, font, font_scale, white, 2, cv::LINE_AA);
            cv::putText(cv_img, label, text_pos, font, font_scale, black, 1, cv::LINE_AA);
        }
        cv::rectangle(cv_img, bbox, black, 3, cv::LINE_8, 0);
        cv::rectangle(cv_img, bbox, color, 2, cv::LINE_8, 0);
        // remove black line between bbox and label
        if (draw_label)
        {
            cv::rectangle(cv_img, hline, color, cv::FILLED, cv::LINE_8);
            if (below)
            {
                cv::rectangle(cv_img, vline, color, cv::FILLED, cv::LINE_8);
            }
        }
    }
}

class webcam_window : public dlib::image_window
{
    public:
    webcam_window()
    {
        update_title();
    }
    bool mirror = true;
    float conf_thresh = 0.25;

    static void print_keyboard_shortcuts()
    {
        std::cout << "Keyboard shortcuts:" << std::endl;
        std::cout << "  h                   Display keyboard shortcuts" << std::endl;
        std::cout << "  m                   Toggle mirror mode" << std::endl;
        std::cout << "  +, k                Increase confidence threshold by 0.05" << std::endl;
        std::cout << "  -, j                Decrease confidence threshold by 0.05" << std::endl;
        std::cout << "  q                   Quit the application" << std::endl;
        std::cout << std::endl;
    }

    private:
    void update_title()
    {
        std::ostringstream sout;
        sout << "YOLO @" << std::setprecision(2) << std::fixed << conf_thresh;
        set_title(sout.str());
    }
    void on_keydown(unsigned long key, bool /*is_printable*/, unsigned long /*state*/) override
    {
        switch (key)
        {
            case 'h':
                print_keyboard_shortcuts();
                break;
            case 'm':
                mirror = !mirror;
                break;
            case '+':
            case 'k':
                conf_thresh = std::min(conf_thresh + 0.01f, 1.0f);
                update_title();
                break;
            case '-':
            case 'j':
                conf_thresh = std::max(conf_thresh - 0.01f, 0.01f);
                update_title();
                break;
            case 'q':
                close_window();
                break;
            default:
                break;
        }
    }
};

#endif // webcam_window_h_INCLUDED

