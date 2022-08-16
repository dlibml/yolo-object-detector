#ifndef model_impl_h_INCLUDED
#define model_impl_h_INCLUDED
#include "model.h"
#include "yolov7.h"

using net_train_type = yolov7::train_type;
using net_infer_type = yolov7::infer_type;

struct model::impl
{
    impl() = default;
    impl(const dlib::yolo_options& options) : train(options), infer(options) {}
    net_train_type train;
    net_infer_type infer;
};

#endif  // model_impl_h_INCLUDED
