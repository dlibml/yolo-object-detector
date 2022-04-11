#include "webcam_window.h"

#include "yolo_logo.h"

#include <dlib/image_io.h>

webcam_window::webcam_window(drawing_options& opts) : opts(opts)
{
    set_background_color(0, 0, 0);
    update_title();
    set_logo();
}

webcam_window::webcam_window(drawing_options& opts, const double conf_thresh)
    : opts(opts),
      conf_thresh(conf_thresh)
{
    set_background_color(0, 0, 0);
    update_title();
    set_logo();
    create_recording_icon();
}

void webcam_window::show_recording_icon()
{
    add_overlay(recording_icon);
}

void webcam_window::print_keyboard_shortcuts()
{
    std::clog << "Keyboard Shortcuts:" << std::endl;
    std::clog << "  +                         increase confidence threshold by 0.01\n";
    std::clog << "  -                         decrease confidence threshold by 0.01\n";
    std::clog << "  c                         toggle confidence display\n";
    std::clog << "  f                         decrease fill transparency\n";
    std::clog << "  g                         increase fill transparency\n";
    std::clog << "  j                         decrease thickness by 1 pixel\n";
    std::clog << "  k                         increase thickness by 1 pixel\n";
    std::clog << "  h                         display keyboard shortcuts\n";
    std::clog << "  l                         toggle label display\n";
    std::clog << "  m                         toggle mirror mode\n";
    std::clog << "  r                         toggle recording (needs --output option)\n";
    std::clog << "  t                         set thickness to 10\n";
    std::clog << "  q                         quit the application\n";
    std::clog << "  w                         toggle weighted thickness\n";
    std::clog << std::endl;
}

void webcam_window::set_logo()
{
    logo = get_yolo_logo();
    set_image(logo);
}

void webcam_window::update_title()
{
    std::ostringstream sout;
    sout << "YOLO @" << std::setprecision(2) << std::fixed << conf_thresh;
    set_title(sout.str());
}

void webcam_window::create_recording_icon()
{
    const auto c = dlib::point(20, 20);
    const auto color = dlib::rgb_pixel(255, 0, 0);
    for (auto r = 0.0; r < 10; r += 0.1)
    {
        recording_icon.emplace_back(c, r, color);
    }
}

void webcam_window::on_keydown(unsigned long key, bool /*is_printable*/, unsigned long /*state*/)
{
    switch (key)
    {
    case '+':
        conf_thresh = std::min(conf_thresh + 0.01f, 1.0f);
        update_title();
        break;
    case '-':
        conf_thresh = std::max(conf_thresh - 0.01f, 0.01f);
        update_title();
        break;
    case 'c':
        opts.draw_confidence = !opts.draw_confidence;
        break;
    case 'f':
        opts.fill = opts.fill > 0 ? opts.fill - 1 : 0;
        break;
    case 'g':
        opts.fill = opts.fill < 255 ? opts.fill + 1 : 255;
        break;
    case 'h':
        print_keyboard_shortcuts();
        break;
    case 'k':
        opts.thickness += 1;
        break;
    case 'j':
        opts.thickness = opts.thickness > 0 ? opts.thickness - 1 : 0;
        break;
    case 'l':
        opts.draw_labels = !opts.draw_labels;
        break;
    case 'm':
        mirror = !mirror;
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
    case 't':
        opts.thickness = 10;
        break;
    case 'q':
        close_window();
        break;
    case 'w':
        opts.weighted = !opts.weighted;
        break;
    default:
        break;
    }
}
