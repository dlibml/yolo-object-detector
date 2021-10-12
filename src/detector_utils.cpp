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

void setup_detector(net_train_type& net, const dlib::yolo_options& options)
{
    // clang-format off
    visit_computational_layers(net, [](dlib::leaky_relu_& l) { l = dlib::leaky_relu_(0.1); });
    visit_computational_layers(dlib::layer<rgpnet::btag4>(net), [](auto& l) { dlib::disable_bias(l); });
    // dlib::visit_computational_layers(net, [](auto& l)
    // {
    //     dlib::set_learning_rate_multiplier(l, 1);
    //     dlib::set_bias_learning_rate_multiplier(l, 1);
    //     dlib::set_weight_decay_multiplier(l, 1);
    //     dlib::set_bias_weight_decay_multiplier(l, 1);
    // });
    // clang-format on
    dlib::disable_duplicative_biases(net);
    const long num_classes = options.labels.size();
    const long num_anchors_1 = options.anchors.at(dlib::tag_id<ytag8>::id).size();
    const long num_anchors_2 = options.anchors.at(dlib::tag_id<ytag16>::id).size();
    const long num_anchors_3 = options.anchors.at(dlib::tag_id<ytag32>::id).size();
    dlib::layer<ytag8, 2>(net).layer_details().set_num_filters(num_anchors_1 * (num_classes + 5));
    dlib::layer<ytag16, 2>(net).layer_details().set_num_filters(num_anchors_2 * (num_classes + 5));
    dlib::layer<ytag32, 2>(net).layer_details().set_num_filters(num_anchors_3 * (num_classes + 5));
}
