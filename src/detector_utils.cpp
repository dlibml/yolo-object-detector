#include "detector_utils.h"

dlib::rectangle_transform preprocess_image(
    const dlib::matrix<dlib::rgb_pixel>& image,
    dlib::matrix<dlib::rgb_pixel>& output,
    const long image_size,
    const bool use_letterbox,
    const long stride)
{
    if (use_letterbox)
        return dlib::rectangle_transform(inv(letterbox_image(image, output, image_size)));

    const double width = image.nc();
    const double height = image.nr();
    const auto scale = image_size / std::max<double>(height, width);
    output.set_size(
        lround(height * scale / stride) * stride,
        lround(width * scale / stride) * stride);
    resize_image(image, output);
    return dlib::point_transform_affine({width / output.nc(), 0, 0, height / output.nr()}, {0, 0});
}

void postprocess_detections(
    const dlib::rectangle_transform& tform,
    std::vector<dlib::yolo_rect>& detections)
{
    for (auto& d : detections)
        d.rect = tform(d.rect);
}
