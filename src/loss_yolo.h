#ifndef loss_yolo_h_INCLUDED
#define loss_yolo_h_INCLUDED

#include <dlib/dnn.h>

// clang-format off

namespace dlib
{
// ----------------------------------------------------------------------------------------

    struct yolo_options2
    {
    public:
        struct anchor_box_details
        {
            anchor_box_details() = default;
            anchor_box_details(unsigned long w, unsigned long h) : width(w), height(h) {}

            unsigned long width = 0;
            unsigned long height = 0;

            friend inline void serialize(const anchor_box_details& item, std::ostream& out)
            {
                int version = 0;
                serialize(version, out);
                serialize(item.width, out);
                serialize(item.height, out);
            }

            friend inline void deserialize(anchor_box_details& item, std::istream& in)
            {
                int version = 0;
                deserialize(version, in);
                deserialize(item.width, in);
                deserialize(item.height, in);
            }
        };

        yolo_options2() = default;

        template <template <typename> class TAG_TYPE>
        void add_anchors(const std::vector<anchor_box_details>& boxes)
        {
            anchors[tag_id<TAG_TYPE>::id] = boxes;
        }

        // map between the stride and the anchor boxes
        std::map<int, std::vector<anchor_box_details>> anchors;
        std::vector<std::string> labels;
        double iou_ignore_threshold = 0.7;
        double iou_anchor_threshold = 1.0;
        test_box_overlap overlaps_nms = test_box_overlap(0.45, 1.0);
        bool classwise_nms = true;
        double lambda_obj = 1.0;
        double lambda_box = 1.0;
        double lambda_cls = 1.0;

    };

    inline void serialize(const yolo_options2& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.anchors, out);
        serialize(item.labels, out);
        serialize(item.iou_ignore_threshold, out);
        serialize(item.iou_anchor_threshold, out);
        serialize(item.classwise_nms, out);
        serialize(item.overlaps_nms, out);
        serialize(item.lambda_obj, out);
        serialize(item.lambda_box, out);
        serialize(item.lambda_cls, out);
    }

    inline void deserialize(yolo_options2& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::yolo_options.");
        deserialize(item.anchors, in);
        deserialize(item.labels, in);
        deserialize(item.iou_ignore_threshold, in);
        deserialize(item.iou_anchor_threshold, in);
        deserialize(item.classwise_nms, in);
        deserialize(item.overlaps_nms, in);
        deserialize(item.lambda_obj, in);
        deserialize(item.lambda_box, in);
        deserialize(item.lambda_cls, in);
    }

    inline std::ostream& operator<<(std::ostream& out, const std::map<int, std::vector<yolo_options2::anchor_box_details>>& anchors)
    {
        // write anchor boxes grouped by tag id
        size_t tag_count = 0;
        for (const auto& i : anchors)
        {
            const auto& tag_id = i.first;
            const auto& details = i.second;
            if (tag_count++ > 0)
                out << ";";
            out << "tag" << tag_id << ":";
            for (size_t a = 0; a < details.size(); ++a)
            {
                out << details[a].width << "x" << details[a].height;
                if (a + 1 < details.size())
                    out << ",";
            }
        }
        return out;
    }

    namespace impl
    {
        template <template <typename> class TAG_TYPE, template <typename> class... TAG_TYPES>
        struct yolo_helper_impl2
        {
            constexpr static size_t tag_count()
            {
                return 1 + yolo_helper_impl2<TAG_TYPES...>::tag_count();
            }

            static void list_tags(std::ostream& out)
            {
                out << "tag" << tag_id<TAG_TYPE>::id << (tag_count() > 1 ? "," : "");
                yolo_helper_impl2<TAG_TYPES...>::list_tags(out);
            }

            template <typename SUBNET>
            static void get_strides(
                const tensor& input_tensor,
                const SUBNET& sub,
                std::vector<std::pair<double, double>>& strides
            )
            {
                yolo_helper_impl2<TAG_TYPE>::get_strides(input_tensor, sub, strides);
                yolo_helper_impl2<TAG_TYPES...>::get_strides(input_tensor, sub, strides);
            }

            template <typename SUBNET>
            static void tensor_to_dets (
                const tensor& input_tensor,
                const SUBNET& sub,
                const long n,
                const yolo_options2& options,
                const double adjust_threshold,
                std::vector<yolo_rect>& dets
            )
            {
                yolo_helper_impl2<TAG_TYPE>::tensor_to_dets(input_tensor, sub, n, options, adjust_threshold, dets);
                yolo_helper_impl2<TAG_TYPES...>::tensor_to_dets(input_tensor, sub, n, options, adjust_threshold, dets);
            }

            template <
                typename const_label_iterator,
                typename SUBNET
            >
            static void tensor_to_loss (
                const tensor& input_tensor,
                const_label_iterator truth,
                SUBNET& sub,
                const long n,
                const yolo_options2& options,
                double& loss
            )
            {
                yolo_helper_impl2<TAG_TYPE>::tensor_to_loss(input_tensor, truth, sub, n, options, loss);
                yolo_helper_impl2<TAG_TYPES...>::tensor_to_loss(input_tensor, truth, sub, n, options, loss);
            }
        };

        template <template <typename> class TAG_TYPE>
        struct yolo_helper_impl2<TAG_TYPE>
        {
            constexpr static size_t tag_count() { return 1; }

            static void list_tags(std::ostream& out) { out << "tag" << tag_id<TAG_TYPE>::id; }

            template <typename SUBNET>
            static void get_strides(
                const tensor& input_tensor,
                const SUBNET& sub,
                std::vector<std::pair<double, double>>& strides
            )
            {
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                strides.emplace_back(stride_x, stride_y);
            }

            template <typename SUBNET>
            static void tensor_to_dets (
                const tensor& input_tensor,
                const SUBNET& sub,
                const long n,
                const yolo_options2& options,
                const double adjust_threshold,
                std::vector<yolo_rect>& dets
            )
            {
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                DLIB_CASSERT(static_cast<size_t>(output_tensor.k()) == anchors.size() * (options.labels.size() + 5));
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                const long num_feats = output_tensor.k() / anchors.size();
                const long num_classes = num_feats - 5;
                const float* const out_data = output_tensor.host();

                for (size_t a = 0; a < anchors.size(); ++a)
                {
                    const long k = a * num_feats;
                    for (long r = 0; r < output_tensor.nr(); ++r)
                    {
                        for (long c = 0; c < output_tensor.nc(); ++c)
                        {
                            const float obj = out_data[tensor_index(output_tensor, n, a * num_feats + 4, r, c)];
                            if (obj > adjust_threshold)
                            {
                                const auto x = out_data[tensor_index(output_tensor, n, k + 0, r, c)] * 2.0 - 0.5;
                                const auto y = out_data[tensor_index(output_tensor, n, k + 1, r, c)] * 2.0 - 0.5;
                                const auto w = out_data[tensor_index(output_tensor, n, k + 2, r, c)];
                                const auto h = out_data[tensor_index(output_tensor, n, k + 3, r, c)];
                                yolo_rect det(centered_drect(dpoint((x + c) * stride_x, (y + r) * stride_y),
                                                             w / (1 - w) * anchors[a].width,
                                                             h / (1 - h) * anchors[a].height));
                                for (long i = 0; i < num_classes; ++i)
                                {
                                    const float conf = obj * out_data[tensor_index(output_tensor, n, a * num_feats + 5 + i, r, c)];
                                    if (conf > adjust_threshold)
                                        det.labels.emplace_back(conf, options.labels[i]);
                                }
                                if (!det.labels.empty())
                                {
                                    std::sort(det.labels.rbegin(), det.labels.rend());
                                    det.detection_confidence = det.labels[0].first;
                                    det.label = det.labels[0].second;
                                    dets.push_back(std::move(det));
                                }
                            }
                        }
                    }
                }
            }

            template <
                typename const_label_iterator,
                typename SUBNET
            >
            static void tensor_to_loss (
                const tensor& input_tensor,
                const_label_iterator truth,
                SUBNET& sub,
                const long n,
                const yolo_options2& options,
                double& loss
            )
            {
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                DLIB_CASSERT(static_cast<size_t>(output_tensor.k()) == anchors.size() * (options.labels.size() + 5));
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                const long num_feats = output_tensor.k() / anchors.size();
                const long num_classes = num_feats - 5;
                const float* const out_data = output_tensor.host();
                tensor& grad = layer<TAG_TYPE>(sub).get_gradient_input();
                DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
                DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
                const rectangle input_rect(input_tensor.nr(), input_tensor.nc());
                float* g = grad.host();

                // Compute the objectness loss for all grid cells
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        for (size_t a = 0; a < anchors.size(); ++a)
                        {
                            const auto k = a * num_feats;
                            const auto x = out_data[tensor_index(output_tensor, n, k + 0, r, c)] * 2 - 0.5;
                            const auto y = out_data[tensor_index(output_tensor, n, k + 1, r, c)] * 2 - 0.5;
                            const auto w = out_data[tensor_index(output_tensor, n, k + 2, r, c)];
                            const auto h = out_data[tensor_index(output_tensor, n, k + 3, r, c)];

                            // The prediction at r, c for anchor a
                            const yolo_rect pred(centered_drect(
                                dpoint((x + c) * stride_x, (y + r) * stride_y),
                                       w / (1 - w) * anchors[a].width,
                                       h / (1 - h) * anchors[a].height
                            ));

                            // Find the best IoU for all ground truth boxes
                            double best_iou = 0;
                            for (const yolo_rect& truth_box : *truth)
                            {
                                if (truth_box.ignore || !input_rect.contains(center(truth_box.rect)))
                                    continue;
                                best_iou = std::max(best_iou, box_intersection_over_union(truth_box.rect, pred.rect));
                            }

                            const auto o_idx = tensor_index(output_tensor, n, a * num_feats + 4, r, c);
                            // Incur loss for the boxes that are below a certain IoU threshold with any truth box
                            if (best_iou < options.iou_ignore_threshold)
                                g[o_idx] = options.lambda_obj * out_data[o_idx];
                        }
                    }
                }

                // Now find the best anchor box for each truth box
                for (const yolo_rect& truth_box : *truth)
                {
                    if (truth_box.ignore || !input_rect.contains(center(truth_box.rect)))
                        continue;
                    const dpoint t_center = dcenter(truth_box);
                    double best_iou = 0;
                    size_t best_a = 0;
                    size_t best_tag_id = 0;
                    running_stats<double> ious;
                    for (const auto& item : options.anchors)
                    {
                        const auto tag_id = item.first;
                        const auto details = item.second;
                        for (size_t a = 0; a < details.size(); ++a)
                        {
                            const yolo_rect anchor(centered_drect(t_center, details[a].width, details[a].height));
                            const double iou = box_intersection_over_union(truth_box.rect, anchor.rect);
                            ious.add(iou);
                            if (iou > best_iou)
                            {
                                best_iou = iou;
                                best_a = a;
                                best_tag_id = tag_id;
                            }
                        }
                    }

                    double iou_anchor_threshold = options.iou_anchor_threshold;
                    // ATSS: Adaptive Training Sample Selection
                    if (options.iou_anchor_threshold == 0)
                        iou_anchor_threshold = ious.mean() + ious.stddev();

                    // std::cout << "truth: " << truth_box.rect.width() << 'x' << truth_box.rect.height() << '\n';
                    // std::cout << stride_x << 'x' << stride_y << " => iou: " << iou_anchor_threshold << '\n';;
                    // for (const auto& item : options.anchors)
                    // {
                    //     const auto details = item.second;
                    //     for (size_t a = 0; a < details.size(); ++a)
                    //     {
                    //         const yolo_rect anchor(centered_drect(t_center, details[a].width, details[a].height));
                    //         const double iou = box_intersection_over_union(truth_box.rect, anchor.rect);
                    //         std::cout << anchor.rect.width() << 'x' << anchor.rect.height() << ": " << iou << '\n';
                    //     }
                    // }
                    // std::cin.get();

                    for (size_t a = 0; a < anchors.size(); ++a)
                    {
                        // Update best anchor if it's from the current stride, and optionally other anchors
                        if ((best_tag_id == tag_id<TAG_TYPE>::id && best_a == a) || iou_anchor_threshold < 1)
                        {

                            // do not update other anchors if they have low IoU
                            if (!(best_tag_id == tag_id<TAG_TYPE>::id && best_a == a))
                            {
                                const yolo_rect anchor(centered_drect(t_center, anchors[a].width, anchors[a].height));
                                const double iou = box_intersection_over_union(truth_box.rect, anchor.rect);
                                if (iou < iou_anchor_threshold)
                                    continue;
                            }
                            const long k = a * num_feats;
                            const long c = t_center.x() / stride_x;
                            const long r = t_center.y() / stride_y;

                            // Get the truth box target values
                            const double tx = t_center.x() / stride_x - c;
                            const double ty = t_center.y() / stride_y - r;
                            const double tw = truth_box.rect.width() / (anchors[a].width + truth_box.rect.width());
                            const double th = truth_box.rect.height() / (anchors[a].height + truth_box.rect.height());

                            // Scale regression error according to the truth size
                            const double scale_box = options.lambda_box * (2 - truth_box.rect.area() / input_rect.area());

                            // Compute the gradient for the box coordinates, and eliminate grid sensitivity
                            const auto x_idx = tensor_index(output_tensor, n, k + 0, r, c);
                            const auto y_idx = tensor_index(output_tensor, n, k + 1, r, c);
                            const auto w_idx = tensor_index(output_tensor, n, k + 2, r, c);
                            const auto h_idx = tensor_index(output_tensor, n, k + 3, r, c);
                            g[x_idx] = scale_box * (out_data[x_idx] * 2.0 - 0.5 - tx);
                            g[y_idx] = scale_box * (out_data[y_idx] * 2.0 - 0.5 - ty);
                            g[w_idx] = scale_box * (out_data[w_idx] - tw);
                            g[h_idx] = scale_box * (out_data[h_idx] - th);

                            // This grid cell should detect an object
                            const auto o_idx = tensor_index(output_tensor, n, k + 4, r, c);
                            g[o_idx] = options.lambda_obj * (out_data[o_idx] - 1);

                            // Compute the classification error
                            for (long k = 0; k < num_classes; ++k)
                            {
                                const auto c_idx = tensor_index(output_tensor, n, k + 5 + k, r, c);
                                if (truth_box.label == options.labels[k])
                                    g[c_idx] = options.lambda_cls * (out_data[c_idx] - 1);
                                else
                                    g[c_idx] = options.lambda_cls * out_data[c_idx];
                            }
                        }
                    }
                }

                // Compute the L2 loss
                loss += length_squared(rowm(mat(grad), n));
            }
        };
    }

    template <template <typename> class... TAG_TYPES>
    class loss_yolo2_
    {
        static void list_tags(std::ostream& out) { impl::yolo_helper_impl2<TAG_TYPES...>::list_tags(out); }

    public:

        typedef std::vector<yolo_rect> training_label_type;
        typedef std::vector<yolo_rect> output_label_type;

        constexpr static size_t tag_count() { return impl::yolo_helper_impl2<TAG_TYPES...>::tag_count(); }

        loss_yolo2_() {};

        loss_yolo2_(const yolo_options2& options) : options(options) { }

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter,
            double adjust_threshold = 0.25
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
            std::vector<yolo_rect> dets_accum;
            std::vector<yolo_rect> final_dets;
            for (long i = 0; i < input_tensor.num_samples(); ++i)
            {
                dets_accum.clear();
                impl::yolo_helper_impl2<TAG_TYPES...>::tensor_to_dets(input_tensor, sub, i, options, adjust_threshold, dets_accum);

                // Do non-max suppression
                std::sort(dets_accum.rbegin(), dets_accum.rend());
                final_dets.clear();
                for (size_t j = 0; j < dets_accum.size(); ++j)
                {
                    if (overlaps_any_box_nms(final_dets, dets_accum[j]))
                        continue;

                    final_dets.push_back(dets_accum[j]);
                }

                *iter++ = std::move(final_dets);
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
        >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
            double loss = 0;
            for (long i = 0; i < input_tensor.num_samples(); ++i)
            {
                impl::yolo_helper_impl2<TAG_TYPES...>::tensor_to_loss(input_tensor, truth, sub, i, options, loss);
                ++truth;
            }
            return loss / input_tensor.num_samples();
        }

        const yolo_options2& get_options() const { return options; }

        void adjust_nms(double iou_thresh, double percent_covered_thresh = 1, bool classwise = true)
        {
            options.overlaps_nms = test_box_overlap(iou_thresh, percent_covered_thresh);
            options.classwise_nms = classwise;
        }

        friend void serialize(const loss_yolo2_& item, std::ostream& out)
        {
            serialize("loss_yolo_", out);
            size_t count = tag_count();
            serialize(count, out);
            serialize(item.options, out);
        }

        friend void deserialize(loss_yolo2_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_yolo_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_yolo_.");
            size_t count = 0;
            deserialize(count, in);
            if (count != tag_count())
                throw serialization_error("Invalid number of detection tags " + std::to_string(count) +
                                          ", while deserializing dlib::loss_yolo_, expecting " +
                                          std::to_string(tag_count()) + "tags instead.");
            deserialize(item.options, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_yolo2_& item)
        {
            out << "loss_yolo\t (";
            const auto& opts = item.options;
            out << tag_count() << " output" << (tag_count() != 1 ? "s" : "") << ":(";
            list_tags(out);
            out << ")";
            out << ", anchor_boxes:(" << opts.anchors << ")";
            out << ", " << opts.labels.size() << " label" << (opts.labels.size() != 1 ? "s" : "") << ":(";
            for (size_t i = 0; i < opts.labels.size(); ++i)
            {
                out << opts.labels[i];
                if (i + 1 < opts.labels.size())
                    out << ",";
            }
            out << ")";
            out << ", iou_ignore_threshold: " << opts.iou_ignore_threshold;
            out << ", iou_anchor_threshold: " << opts.iou_anchor_threshold;
            out << ", lambda_obj:" << opts.lambda_obj;
            out << ", lambda_box:" << opts.lambda_box;
            out << ", lambda_cls:" << opts.lambda_cls;
            out << ", overlaps_nms:(" << opts.overlaps_nms.get_iou_thresh() << "," << opts.overlaps_nms.get_percent_covered_thresh() << ")";
            out << ", classwise_nms:" << std::boolalpha << opts.classwise_nms;
            out << ")";
            return out;
        }

        friend void to_xml(const loss_yolo2_& /*item*/, std::ostream& out)
        {
            out << "<loss_yolo/>";
        }

    private:

        yolo_options2 options;

        inline bool overlaps_any_box_nms (
            const std::vector<yolo_rect>& boxes,
            const yolo_rect& box
        ) const
        {
            for (const auto& b : boxes)
            {
                if (options.overlaps_nms(b.rect, box.rect))
                {
                    if (options.classwise_nms)
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
    };

    template <template <typename> class TAG_1, template <typename> class TAG_2, template <typename> class TAG_3, typename SUBNET>
    using loss_yolo2 = add_loss_layer<loss_yolo2_<TAG_1, TAG_2, TAG_3>, SUBNET>;
}
// clang-format on

#endif  // loss_yolo_h_INCLUDED
