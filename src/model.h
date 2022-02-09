#ifndef model_h_INCLUDED
#define model_h_INCLUDED

#include "yolor.h"

using net_train_type = yolor::train_type;
using net_infer_type = yolor::infer_type;

template <typename SUBNET> using ytag3 = yolor::ytag3<SUBNET>;
template <typename SUBNET> using ytag4 = yolor::ytag4<SUBNET>;
template <typename SUBNET> using ytag5 = yolor::ytag5<SUBNET>;
template <typename SUBNET> using ytag6 = yolor::ytag6<SUBNET>;

class model_train : public net_train_type
{
    public:
    model_train();
    model_train(const dlib::yolo_options& options);
    void save(const std::string& path);
    void load(const std::string& path);
};

class model_infer : public net_infer_type
{
    public:
    model_infer();
    model_infer(const net_train_type& net);
    void save(const std::string& path);
    void load(const std::string& path);
    void fuse();
    void print_loss_details() const;
};

#endif  // model_h_INCLUDED
