#ifndef utils_h_INCLUDED
#define utils_h_INCLUDED

#include "model.h"

#include <dlib/image_transforms.h>

template <typename T, typename U> bool overlaps_any_box(
    const std::vector<T>& boxes,
    const U& box,
    const dlib::test_box_overlap& overlaps = dlib::test_box_overlap(0.45, 1),
    const bool classwise = true)
{
    for (const auto& b : boxes)
    {
        if (overlaps(b.rect, box.rect))
        {
            if (classwise)
            {
                if (b.label == box.label)
                    return true;
            }
            else
            {
                return true;
            }
        }
    }
    return false;
}

dlib::rectangle_transform preprocess_image(
    const dlib::matrix<dlib::rgb_pixel>& image,
    dlib::matrix<dlib::rgb_pixel>& output,
    const long image_size,
    const bool use_letterbox = true,
    const long stride = 32
);

void postprocess_detections(
    const dlib::rectangle_transform& tform,
    std::vector<dlib::yolo_rect>& detections);

#endif  // utils_h_INCLUDED
