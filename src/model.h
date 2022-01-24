#ifndef model_h_INCLUDED
#define model_h_INCLUDED

#include "yolor.h"

using net_train_type = yolor::train_type;
using net_infer_type = yolor::infer_type;

template <typename SUBNET> using ytag3 = yolor::ytag3<SUBNET>;
template <typename SUBNET> using ytag4 = yolor::ytag4<SUBNET>;
template <typename SUBNET> using ytag5 = yolor::ytag5<SUBNET>;
template <typename SUBNET> using ytag6 = yolor::ytag6<SUBNET>;

#endif  // model_h_INCLUDED
