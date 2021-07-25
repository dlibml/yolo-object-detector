#ifndef model_h_INCLUDED
#define model_h_INCLUDED

#include "rgpnet.h"

using net_train_type = rgpnet::train;
using net_infer_type = rgpnet::infer;

template <typename SUBNET> using ytag8 = rgpnet::ytag8<SUBNET>;
template <typename SUBNET> using ytag16 = rgpnet::ytag16<SUBNET>;
template <typename SUBNET> using ytag32 = rgpnet::ytag32<SUBNET>;

struct model_train
{
    model_train() = default;
    model_train(const dlib::yolo_options& options);

    dlib::dnn_trainer<net_train_type, dlib::sgd> get_trainer(
        float weight_decay = 0.0005,
        float momentum = 0.9,
        const std::vector<int>& gpus = {0});

    net_train_type net;
};

struct model_infer
{
    model_infer() = default;
    model_infer(const dlib::yolo_options& options);

    dlib::dnn_trainer<net_infer_type, dlib::sgd> get_trainer(
        float weight_decay = 0.0005,
        float momentum = 0.9,
        const std::vector<int>& gpus = {0});

    net_infer_type net;
};

#endif  // model_h_INCLUDED
