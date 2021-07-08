#ifndef model_h_INCLUDED
#define model_h_INCLUDED

#include "rgpnet.h"

struct model_train
{
    model_train() = default;
    model_train(const dlib::yolo_options& options);
    rgpnet::train net;
    dlib::dnn_trainer<rgpnet::train, dlib::sgd>
        get_trainer(float weight_decay = 0.0005, float momentum = 0.9);
};

struct model_infer
{
    model_infer() = default;
    model_infer(const dlib::yolo_options& options);
    rgpnet::infer net;
    dlib::dnn_trainer<rgpnet::infer, dlib::sgd>
        get_trainer(float weight_decay = 0.0005, float momentum = 0.9);
};

#endif  // model_h_INCLUDED
