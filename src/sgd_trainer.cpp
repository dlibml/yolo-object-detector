
#include "sgd_trainer.h"

sgd_trainer::sgd_trainer(
    net_train_type& net,
    const std::vector<int> gpus,
    const float weight_decay,
    const float momentum)
    : dlib::dnn_trainer<net_train_type, dlib::sgd>(net, dlib::sgd(weight_decay, momentum), gpus)
{
}

aux_trainer::aux_trainer(net_infer_type& net) : dlib::dnn_trainer<net_infer_type>(net)
{
}
