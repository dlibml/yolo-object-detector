#include "model.h"

model_train::model_train(const dlib::yolo_options& options) : net(options)
{
}

dlib::dnn_trainer<net_train_type, dlib::sgd>
    model_train::get_trainer(float weight_decay, float momentum, const std::vector<int>& gpus)
{
    return dlib::dnn_trainer<net_train_type, dlib::sgd>(
        net,
        dlib::sgd(weight_decay, momentum),
        gpus);
}

model_infer::model_infer(const dlib::yolo_options& options) : net(options)
{
}

dlib::dnn_trainer<net_infer_type, dlib::sgd>
    model_infer::get_trainer(float weight_decay, float momentum, const std::vector<int>& gpus)
{
    return dlib::dnn_trainer<net_infer_type, dlib::sgd>(
        net,
        dlib::sgd(weight_decay, momentum),
        gpus);
}
