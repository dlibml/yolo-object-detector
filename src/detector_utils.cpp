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
    using namespace dlib;
    visit_computational_layers(net, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
    visit_computational_layers(net, [](auto& l) { disable_bias(l); });
    // re-enable the biases in the convolutions for YOLO layers
    layer<ytag3, 2>(net).layer_details().enable_bias();
    layer<ytag4, 2>(net).layer_details().enable_bias();
    layer<ytag5, 2>(net).layer_details().enable_bias();
    layer<ytag6, 2>(net).layer_details().enable_bias();
    // set the number of filters in the convolutions for YOLO layers
    const long num_classes = options.labels.size();
    const long num_anchors_p3 = options.anchors.at(tag_id<ytag3>::id).size();
    const long num_anchors_p4 = options.anchors.at(tag_id<ytag4>::id).size();
    const long num_anchors_p5 = options.anchors.at(tag_id<ytag5>::id).size();
    const long num_anchors_p6 = options.anchors.at(tag_id<ytag6>::id).size();
    layer<ytag3, 2>(net).layer_details().set_num_filters(num_anchors_p3 * (num_classes + 5));
    layer<ytag4, 2>(net).layer_details().set_num_filters(num_anchors_p4 * (num_classes + 5));
    layer<ytag5, 2>(net).layer_details().set_num_filters(num_anchors_p5 * (num_classes + 5));
    layer<ytag6, 2>(net).layer_details().set_num_filters(num_anchors_p6 * (num_classes + 5));
    set_all_bn_running_stats_window_sizes(net, 1000);
}

void print_loss_details(const net_infer_type& net)
{
    const auto& opts = net.loss_details().get_options();
    std::cout << "YOLO loss details (" << opts.anchors.size() << " outputs)" << '\n';
    std::cout << "  anchors:\n";
    for (const auto& [tag_id, anchors] : opts.anchors)
    {
        std::cout << "    " << tag_id << ": ";
        for (size_t i = 0; i < anchors.size(); ++i)
        {
            std::cout << anchors[i].width << 'x' << anchors[i].height;
            if (i + 1 < anchors.size())
                std::cout << ", ";
        }
        std::cout << '\n';
    }
    std::cout << "  iou_ignore_threshold: " << opts.iou_ignore_threshold << '\n';
    std::cout << "  iou_anchor_threshold: " << opts.iou_anchor_threshold << '\n';
    std::cout << "  lambda_obj: " << opts.lambda_obj << '\n';
    std::cout << "  lambda_box: " << opts.lambda_box << '\n';
    std::cout << "  lambda_cls: " << opts.lambda_cls << '\n';
    std::cout << "  overlaps_nms: (" << opts.overlaps_nms.get_iou_thresh() << ", "
              << opts.overlaps_nms.get_percent_covered_thresh() << ")" << '\n';
    std::cout << "  classwise_nms: " << std::boolalpha << opts.classwise_nms << '\n';
    std::cout << "  " << opts.labels.size() << " labels:\n";
    for (size_t i = 0; i < opts.labels.size(); ++i)
        std::cout << "    " << std::setw(2) << i << ". " << opts.labels[i] << '\n';
    std::cout << '\n';
}
