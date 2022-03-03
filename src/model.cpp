#include "model.h"

#include "model_impl.h"

using namespace dlib;

model::~model() = default;

model::model() : pimpl(std::make_unique<model::impl>())
{
}

model::model(const yolo_options& options) : pimpl(std::make_unique<model::impl>(options))
{
    auto& net = pimpl->train;
    using namespace dlib;
    // setup the leaky relu activations
    visit_computational_layers(net, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
    disable_duplicative_biases(net);
    // set the number of filters in the convolutions for YOLO layers
    const long num_classes = options.labels.size();
    const long num_anchors_p3 = options.anchors.at(tag_id<ytag3>::id).size();
    const long num_anchors_p4 = options.anchors.at(tag_id<ytag4>::id).size();
    const long num_anchors_p5 = options.anchors.at(tag_id<ytag5>::id).size();
    // const long num_anchors_p6 = options.anchors.at(tag_id<ytag6>::id).size();
    layer<ytag3, 2>(net).layer_details().set_num_filters(num_anchors_p3 * (num_classes + 5));
    layer<ytag4, 2>(net).layer_details().set_num_filters(num_anchors_p4 * (num_classes + 5));
    layer<ytag5, 2>(net).layer_details().set_num_filters(num_anchors_p5 * (num_classes + 5));
    // increase the batch normalization window size
    set_all_bn_running_stats_window_sizes(net, 1000);
}

void model::sync()
{
    pimpl->infer = pimpl->train;
}

void model::clean()
{
    pimpl->train.clean();
    pimpl->infer.clean();
}

void model::save_train(const std::string& path)
{
    pimpl->train.clean();
    serialize(path) << pimpl->train;
}

void model::load_train(const std::string& path)
{
    deserialize(path) >> pimpl->train;
}

void model::save_infer(const std::string& path)
{
    pimpl->infer.clean();
    serialize(path) << pimpl->infer;
}

void model::load_infer(const std::string& path)
{
    deserialize(path) >> pimpl->infer;
}

void model::load_backbone(const std::string& path)
{
    net_train_type temp;
    deserialize(path) >> temp;
    layer<ytag3, 3>(pimpl->train) = layer<ytag3, 3>(temp);
    sync();
}

auto model::operator()(const matrix<rgb_pixel>& image, const float conf) -> std::vector<yolo_rect>
{
    return pimpl->infer.process(image, conf);
}

auto model::operator()(
    const std::vector<matrix<rgb_pixel>>& images,
    const size_t batch_size,
    const float conf) -> std::vector<std::vector<yolo_rect>>
{
    return pimpl->infer.process_batch(images, batch_size, conf);
}

void model::adjust_nms(const float iou_threshold, const float ratio_covered, const bool classwise)
{
    pimpl->train.loss_details().adjust_nms(iou_threshold, ratio_covered, classwise);
    pimpl->infer.loss_details().adjust_nms(iou_threshold, ratio_covered, classwise);
}

void model::fuse()
{
    fuse_layers(pimpl->infer);
}

const yolo_options& model::get_options() const
{
    return pimpl->infer.loss_details().get_options();
}

void model::print(std::ostream& out) const
{
    out << pimpl->train << '\n';
}

void model::print_loss_details() const
{
    const auto& opts = pimpl->infer.loss_details().get_options();
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
