#include "detector_utils.h"

dlib::rectangle_transform preprocess_image(
    const dlib::matrix<dlib::rgb_pixel>& image,
    dlib::matrix<dlib::rgb_pixel>& output,
    const long image_size)
{
    return dlib::rectangle_transform(inv(letterbox_image(image, output, image_size)));
}

void postprocess_detections(
    const dlib::rectangle_transform& tform,
    std::vector<dlib::yolo_rect>& detections)
{
    for (auto& d : detections)
        d.rect = tform(d.rect);
}
