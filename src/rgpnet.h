#ifndef RGPNet_H
#define RGPNet_H

#include "vovnet.h"

namespace rgpnet
{
    // clang-format off
    using namespace dlib;

    // The YOLO tags
    template <typename SUBNET> using ytag8 = add_tag_layer<4008, SUBNET>;
    template <typename SUBNET> using ytag16 = add_tag_layer<4016, SUBNET>;
    template <typename SUBNET> using ytag32 = add_tag_layer<4032, SUBNET>;

    // The backbone tags
    template <typename SUBNET> using btag1 = add_tag_layer<8001, SUBNET>;
    template <typename SUBNET> using btag2 = add_tag_layer<8002, SUBNET>;
    template <typename SUBNET> using btag3 = add_tag_layer<8003, SUBNET>;
    template <typename SUBNET> using bskip1 = add_skip_layer<btag1, SUBNET>;
    template <typename SUBNET> using bskip2 = add_skip_layer<btag2, SUBNET>;
    template <typename SUBNET> using bskip3 = add_skip_layer<btag3, SUBNET>;

    // RPGNet number of filters at each level
    static const long feats1 = 64;
    static const long feats2 = 128;
    static const long feats3 = 256;
    static const long feats4 = 512;

    // Custom tags to refer to the processed inputs from the backbone
    template <typename SUBNET> using itag1 = add_tag_layer<5001, SUBNET>;
    template <typename SUBNET> using itag2 = add_tag_layer<5002, SUBNET>;
    template <typename SUBNET> using itag3 = add_tag_layer<5003, SUBNET>;
    template <typename SUBNET> using itag4 = add_tag_layer<5004, SUBNET>;

    // Custom tags for the adaptor modules
    template <typename SUBNET> using atag = add_tag_layer<7000, SUBNET>;
    template <typename SUBNET> using atag4 = add_tag_layer<7004, SUBNET>;
    template <typename SUBNET> using atag3 = add_tag_layer<7003, SUBNET>;
    template <typename SUBNET> using atag2 = add_tag_layer<7002, SUBNET>;
    template <typename SUBNET> using askip2 = add_skip_layer<atag2, SUBNET>;
    template <typename SUBNET> using askip3 = add_skip_layer<atag3, SUBNET>;
    template <typename SUBNET> using askip4 = add_skip_layer<atag4, SUBNET>;

    // BN is bn_con or affine and ACT is an activation layer, such as relu or mish
    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        // ----------------------------- VoVNet Backbone ----------------------------- //

        // the resnet basic block, where BN is bn_con or affine
        template<long num_filters, int stride, typename SUBNET>
        using basicblock = BN<con<num_filters, 3, 3, 1, 1,
                ACT<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

        // the resnet residual, where BLOCK is either basicblock or bottleneck
        template<template<long, int, typename> class BLOCK, long num_filters, typename SUBNET>
        using residual = add_prev1<BLOCK<num_filters, 1, tag1<SUBNET>>>;

        // residual block with optional downsampling
        template<
            template<template<long, int, typename> class, long, typename> class RESIDUAL,
            template<long, int, typename> class BLOCK,
            long num_filters,
            typename SUBNET
        >
        using residual_block = ACT<RESIDUAL<BLOCK, num_filters, SUBNET>>;

        template <long num_filters, typename SUBNET>
        using resbasicblock = residual_block<residual, basicblock, num_filters, SUBNET>;

        template <typename INPUT>
        using backbone = typename vovnet::def<ACT, BN>::template osa_module3<512, 112,
                         typename vovnet::def<ACT, BN>::template maxpool<
                   btag3<typename vovnet::def<ACT, BN>::template osa_module3<384, 96,
                         typename vovnet::def<ACT, BN>::template maxpool<
                   btag2<typename vovnet::def<ACT, BN>::template osa_module3<256, 80,
                         typename vovnet::def<ACT, BN>::template maxpool<
                   btag1<typename vovnet::def<ACT, BN>::template osa_module3<112, 64,
                         typename vovnet::def<ACT, BN>::template stem<INPUT>>>>>>>>>>>;

        // --------------------------------- RGPNet --------------------------------- //

        template <long num_filters, typename SUBNET>
        using downsampler = add_layer<con_<num_filters, 3, 3, 2, 2, 1, 1>, SUBNET>;
        template <long num_filters, typename SUBNET>
        // using upsampler = cont<num_filters, 2, 2, 2, 2, SUBNET>;
        using upsampler = con<num_filters, 1, 1, 1, 1, upsample<2, SUBNET>>;

        // template <template <typename> class TAGGED, typename SUBNET>
        // using resize_and_add = add_layer<add_prev_<TAGGED>, resize_prev_to_tagged<TAGGED, SUBNET>>;
        // using resize_and_add = add_layer<add_prev_<TAGGED>,  SUBNET>;

        // The processed backbone levels that serve as the input of RGPNet
        template <typename SUBNET> using in_lvl1 = ACT<BN<con<feats1, 1, 1, 1, 1, bskip1<SUBNET>>>>;
        template <typename SUBNET> using in_lvl2 = ACT<BN<con<feats2, 1, 1, 1, 1, bskip2<SUBNET>>>>;
        template <typename SUBNET> using in_lvl3 = ACT<BN<con<feats3, 1, 1, 1, 1, bskip3<SUBNET>>>>;
        template <typename SUBNET> using in_lvl4 = ACT<BN<con<feats4, 1, 1, 1, 1, SUBNET>>>;

        // The and downsampled versions of the backbone processed levels
        template <typename SUBNET> using in_lvl1d = downsampler<feats2, itag1<in_lvl1<SUBNET>>>;
        template <typename SUBNET> using in_lvl2d = downsampler<feats3, itag2<in_lvl2<SUBNET>>>;
        template <typename SUBNET> using in_lvl3d = downsampler<feats4, itag3<in_lvl3<SUBNET>>>;

        // adaptor4 adds in_lvl4 and in_lvl3d
        template <typename SUBNET> using adaptor4 =
        resbasicblock<feats4, add_prev<itag4, in_lvl3d<itag4<in_lvl4<SUBNET>>>>>;

        // adaptor3 adds in_lvl3, inlvl2d and in_lvl4u
        template <typename SUBNET> using adaptor3 =
        resbasicblock<feats3, add_prev<itag3,
        add_prev<atag, in_lvl2d<atag<upsampler<feats3, SUBNET>>>>>>;

        // adaptor2 adds in_lvl2, in_lvl1d, and inlvl3u
        template <typename SUBNET> using adaptor2 =
        resbasicblock<feats2, add_prev<itag2,
        add_prev<atag, in_lvl1d<atag<upsampler<feats2, SUBNET>>>>>>;

        // adaptor1 adds in_lvl1 and inlvl2u
        template <typename SUBNET> using adaptor1 =
        resbasicblock<feats1, add_prev<itag1, upsampler<feats1, SUBNET>>>;

        template <typename SUBNET>
        using spp = concat4<tag4, tag3, tag2, tag1, // 113
               tag4<max_pool<13, 13, 1, 1,          // 112
                    skip1<                          // 111
               tag3<max_pool<9, 9, 1, 1,            // 110
                    skip1<                          // 109
               tag2<max_pool<5, 5, 1, 1,            // 108
               tag1<SUBNET>>>>>>>>>>;

        // The RGPNet type definition
        using net_type = loss_yolo<ytag8, ytag16, ytag32,
        // using net_type = loss_multiclass_log_per_pixel<
        ytag8<sig<con<1, 1, 1, 1, 1, adaptor2<askip3<
        ytag16<sig<con<1, 1, 1, 1, 1, atag3<adaptor3<askip4<
        ytag32<sig<con<1, 1, 1, 1, 1, atag4<adaptor4<
        spp<backbone<input_rgb_image>>>>>>>>>>>>>>>>>>>;
    };

    using train = def<relu, bn_con>::net_type;
    using infer = def<relu, affine>::net_type;

    // clang-format on
}  // namespace rgpnet

#endif  // RGPNet_H
