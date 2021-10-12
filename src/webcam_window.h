#ifndef webcam_window_h_INCLUDED
#define webcam_window_h_INCLUDED

#include <dlib/gui_widgets.h>

class webcam_window : public dlib::image_window
{
    public:
    webcam_window() { update_title(); };
    explicit webcam_window(const double conf_thresh) : conf_thresh(conf_thresh)
    {
        update_title();
        create_recording_icon();
    }
    bool mirror = true;
    float conf_thresh = 0.25;
    bool recording = false;
    bool can_record = false;

    static void print_keyboard_shortcuts()
    {
        std::cout << "Keyboard Shortcuts:" << std::endl;
        std::cout << "  h                         display keyboard shortcuts\n";
        std::cout << "  m                         toggle mirror mode\n";
        std::cout << "  +, k                      increase confidence threshold by 0.01\n";
        std::cout << "  -, j                      decrease confidence threshold by 0.01\n";
        std::cout << "  r                         toggle recording (needs --output option)\n";
        std::cout << "  q                         quit the application\n";
        std::cout << std::endl;
    }

    void show_recording_icon() { add_overlay(recording_icon); }

    private:
    std::vector<overlay_circle> recording_icon;
    void update_title()
    {
        std::ostringstream sout;
        sout << "YOLO @" << std::setprecision(2) << std::fixed << conf_thresh;
        set_title(sout.str());
    }
    void create_recording_icon()
    {
        const auto c = dlib::point(20, 20);
        const auto color = dlib::rgb_pixel(255, 0, 0);
        for (auto r = 0.0; r < 10; r += 0.1)
        {
            recording_icon.emplace_back(c, r, color);
        }
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
        case 'r':
            if (not can_record)
                break;
            recording = !recording;
            if (recording)
                show_recording_icon();
            else
                clear_overlay();
            break;
        case 'q':
            close_window();
            break;
        default:
            break;
        }
    }
};

#endif  // webcam_window_h_INCLUDED
