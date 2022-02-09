#include "model.h"

model_train::model_train() = default;

model_train::model_train(const dlib::yolo_options& options) : net_train_type(options)
{
    using namespace dlib;
    visit_computational_layers(*this, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
    visit_computational_layers(*this, [](auto& l) { disable_bias(l); });
    // re-enable the biases in the convolutions for YOLO layers
    layer<ytag3, 2>(*this).layer_details().enable_bias();
    layer<ytag4, 2>(*this).layer_details().enable_bias();
    layer<ytag5, 2>(*this).layer_details().enable_bias();
    layer<ytag6, 2>(*this).layer_details().enable_bias();
    // set the number of filters in the convolutions for YOLO layers
    const long num_classes = options.labels.size();
    const long num_anchors_p3 = options.anchors.at(tag_id<ytag3>::id).size();
    const long num_anchors_p4 = options.anchors.at(tag_id<ytag4>::id).size();
    const long num_anchors_p5 = options.anchors.at(tag_id<ytag5>::id).size();
    const long num_anchors_p6 = options.anchors.at(tag_id<ytag6>::id).size();
    layer<ytag3, 2>(*this).layer_details().set_num_filters(num_anchors_p3 * (num_classes + 5));
    layer<ytag4, 2>(*this).layer_details().set_num_filters(num_anchors_p4 * (num_classes + 5));
    layer<ytag5, 2>(*this).layer_details().set_num_filters(num_anchors_p5 * (num_classes + 5));
    layer<ytag6, 2>(*this).layer_details().set_num_filters(num_anchors_p6 * (num_classes + 5));
    set_all_bn_running_stats_window_sizes(*this, 1000);
}

void model_train::save(const std::string& path)
{
    clean();
    dlib::serialize(path) << *this;
}

void model_train::load(const std::string& path)
{
    dlib::deserialize(path) >> *this;
}

model_infer::model_infer() = default;

model_infer::model_infer(const net_train_type& net) : net_infer_type(net)
{
}

void model_infer::save(const std::string& path)
{
    clean();
    dlib::serialize(path) << *this;
}

void model_infer::load(const std::string& path)
{
    dlib::deserialize(path) >> *this;
}

void model_infer::fuse()
{
    fuse_layers(*this);
}

void model_infer::print_loss_details() const
{
    const auto& opts = loss_details().get_options();
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
