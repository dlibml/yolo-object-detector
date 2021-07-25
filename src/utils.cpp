#include "utils.h"

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

void setup_detector(net_train_type& net, const dlib::yolo_options& options)
{
    dlib::disable_duplicative_biases(net);
    const long num_classes = options.labels.size();
    const long num_anchors_1 = options.anchors.at(dlib::tag_id<ytag8>::id).size();
    const long num_anchors_2 = options.anchors.at(dlib::tag_id<ytag16>::id).size();
    const long num_anchors_3 = options.anchors.at(dlib::tag_id<ytag32>::id).size();
    dlib::layer<ytag8, 2>(net).layer_details().set_num_filters(num_anchors_1 * (num_classes + 5));
    dlib::layer<ytag16, 2>(net).layer_details().set_num_filters(num_anchors_2 * (num_classes + 5));
    dlib::layer<ytag32, 2>(net).layer_details().set_num_filters(num_anchors_3 * (num_classes + 5));
}
