#ifndef model_impl_h_INCLUDED
#define model_impl_h_INCLUDED
#include "model.h"
#include "yolor.h"

using net_train_type = yolor::train_type;
using net_infer_type = yolor::infer_type;

struct model::impl
{
    impl() = default;
    impl(const dlib::yolo_options& options) : train(options), infer(options) {}
    net_train_type train;
    net_infer_type infer;
};

#endif  // model_impl_h_INCLUDED
