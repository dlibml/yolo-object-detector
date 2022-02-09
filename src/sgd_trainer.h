#ifndef trainer_h_INCLUDED
#define trainer_h_INCLUDED

#include "model.h"

class sgd_trainer : public dlib::dnn_trainer<net_train_type, dlib::sgd>
{
    public:
    sgd_trainer() = delete;
    sgd_trainer(
        net_train_type& net,
        const std::vector<int> gpus = {0},
        const float weight_decay = 0.0005,
        const float momentum = 0.9);
};

class aux_trainer : public dlib::dnn_trainer<net_infer_type>
{
    public:
    aux_trainer(net_infer_type& net);
};

#endif  // trainer_h_INCLUDED
