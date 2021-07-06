#ifndef utils_h_INCLUDED
#define utils_h_INCLUDED

#include <dlib/image_transforms.h>
#include "rgpnet.h"

dlib::rectangle_transform preprocess_image(
    const dlib::matrix<dlib::rgb_pixel>& image,
    dlib::matrix<dlib::rgb_pixel>& output,
    const long image_size);

void postprocess_detections(
    const dlib::rectangle_transform& tform,
    std::vector<dlib::yolo_rect>& detections);

void setup_detector(rgpnet::train& net, const dlib::yolo_options& options);

#endif  // utils_h_INCLUDED
