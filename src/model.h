#ifndef model_h_INCLUDED
#define model_h_INCLUDED

#include "rgpnet.h"

using net_train_type = rgpnet::train;
using net_infer_type = rgpnet::infer;

template <typename SUBNET> using ytag8 = rgpnet::ytag8<SUBNET>;
template <typename SUBNET> using ytag16 = rgpnet::ytag16<SUBNET>;
template <typename SUBNET> using ytag32 = rgpnet::ytag32<SUBNET>;
template <typename SUBNET> using ytag64 = rgpnet::ytag64<SUBNET>;

#endif  // model_h_INCLUDED
