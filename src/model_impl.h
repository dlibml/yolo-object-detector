#ifndef model_impl_h_INCLUDED
#define model_impl_h_INCLUDED
#include "model.h"
#include "yolov5.h"

using net_train_type = yolov5::train_type_l;
using net_infer_type = yolov5::infer_type_l;

struct model::impl
{
    impl() = default;
    impl(const dlib::yolo_options& options) : train(options), infer(options) {}
    net_train_type train;
    net_infer_type infer;
};

#endif  // model_impl_h_INCLUDED
