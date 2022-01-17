#ifndef webcam_window_h_INCLUDED
#define webcam_window_h_INCLUDED

#include <dlib/gui_widgets.h>

class webcam_window : public dlib::image_window
{
    public:
    webcam_window();
    explicit webcam_window(const double conf_thresh);
    static void print_keyboard_shortcuts();
    void show_recording_icon();

    bool mirror = true;
    float conf_thresh = 0.25;
    bool recording = false;
    bool can_record = false;

    private:
    void set_logo();
    void update_title();
    void create_recording_icon();
    void on_keydown(unsigned long key, bool /*is_printable*/, unsigned long /*state*/) override;

    std::vector<overlay_circle> recording_icon;
    dlib::matrix<dlib::rgb_alpha_pixel> logo;
};

#endif  // webcam_window_h_INCLUDED
