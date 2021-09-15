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
    template <typename SUBNET> using btag4 = add_tag_layer<8004, SUBNET>;
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
    template <typename SUBNET> using atag0 = add_tag_layer<7000, SUBNET>;
    template <typename SUBNET> using atag1 = add_tag_layer<7001, SUBNET>;
    template <typename SUBNET> using atag2 = add_tag_layer<7002, SUBNET>;
    template <typename SUBNET> using atag3 = add_tag_layer<7003, SUBNET>;
    template <typename SUBNET> using atag4 = add_tag_layer<7004, SUBNET>;
    template <typename SUBNET> using askip2 = add_skip_layer<atag2, SUBNET>;
    template <typename SUBNET> using askip3 = add_skip_layer<atag3, SUBNET>;
    template <typename SUBNET> using askip4 = add_skip_layer<atag4, SUBNET>;

    // BN is bn_con or affine and ACT is an activation layer, such as relu or mish
    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long out_filters, long in_filters,typename SUBNET>
        using osa_module = typename vovnet::def<ACT, BN>::template osa_module5<out_filters, in_filters, SUBNET>;
        template <typename SUBNET> using stem = typename vovnet::def<ACT, BN>::template stem<SUBNET>;
        template <typename SUBNET> using maxpool = typename vovnet::def<ACT, BN>::template maxpool<SUBNET>;
        template <typename SUBNET> using id_mapping = vovnet::id_mapping<SUBNET>;
        template <typename SUBNET> using osa_module_64 = osa_module<64, 64, SUBNET>;
        template <typename SUBNET> using osa_module_128 = osa_module<128, 96, SUBNET>;
        template <typename SUBNET> using osa_module_256 = osa_module<256, 128, SUBNET>;
        template <typename SUBNET> using osa_module_512 = osa_module<512, 160, SUBNET>;
        template <typename SUBNET> using osa_module_768 = osa_module<768, 192, SUBNET>;
        template <typename SUBNET> using osa_module_1024 = osa_module<1024, 224, SUBNET>;
        template <typename SUBNET> using osa_module_id_512 = id_mapping<osa_module_512<SUBNET>>;
        template <typename SUBNET> using osa_module_id_768 = id_mapping<osa_module_768<SUBNET>>;
        template <typename SUBNET> using osa_module_id_1024 = id_mapping<osa_module_1024<SUBNET>>;

        template <typename INPUT>
        using backbone_57 = repeat<2, osa_module_id_1024, osa_module_1024<
                            maxpool<
                      btag3<repeat<3, osa_module_id_768, osa_module_768<
                            maxpool<
                      btag2<osa_module_512<
                            maxpool<
                      btag1<osa_module_256<
                            stem<INPUT>>>>>>>>>>>>>;

        // --------------------------------- RGPNet --------------------------------- //

        template <long num_filters, typename SUBNET>
        using downsampler = ACT<BN<add_layer<con_<num_filters, 3, 3, 2, 2, 1, 1>, SUBNET>>>;
        template <long num_filters, typename SUBNET>
        using upsampler = ACT<BN<add_layer<cont_<num_filters, 2, 2, 2, 2, 0, 0>, SUBNET>>>;
        // using upsampler = upsample<2, con<num_filters, 1, 1, 1, 1, SUBNET>>;

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
        osa_module_512<concat2<itag4, atag0, atag0<in_lvl3d<itag4<in_lvl4<SUBNET>>>>>>;

        // adaptor3 adds in_lvl3, inlvl2d and in_lvl4u
        template <typename SUBNET> using adaptor3 =
        osa_module_256<concat3<itag3, atag0, atag1, atag1<in_lvl2d<atag0<upsampler<feats3, SUBNET>>>>>>;

        // adaptor2 adds in_lvl2, in_lvl1d, and inlvl3u
        template <typename SUBNET> using adaptor2 =
        osa_module_128<concat3<itag2, atag0, atag1, atag1<in_lvl1d<atag0<upsampler<feats2, SUBNET>>>>>>;

        // adaptor1 adds in_lvl1 and inlvl2u
        template <typename SUBNET> using adaptor1 =
        osa_module_64<concat2<itag1, atag0, atag0<upsampler<feats1, SUBNET>>>>;

        template <typename SUBNET>
        using spp = con<512, 1, 1, 1, 1,
                    concat4<tag4, tag3, tag2, tag1,
               tag4<max_pool<13, 13, 1, 1,
                    skip1<
               tag3<max_pool<9, 9, 1, 1,
                    skip1<
               tag2<max_pool<5, 5, 1, 1,
               tag1<SUBNET>>>>>>>>>>>;

        template <typename INPUT>
        using backbone = btag4<backbone_57<INPUT>>;

        using net_type = loss_yolo<ytag8, ytag16, ytag32,
        ytag8<sig<con<1, 1, 1, 1, 1, adaptor2<askip3<
        ytag16<sig<con<1, 1, 1, 1, 1, atag3<adaptor3<askip4<
        ytag32<sig<con<1, 1, 1, 1, 1, atag4<adaptor4<
        spp<backbone<input_rgb_image>>>>>>>>>>>>>>>>>>>;
    };

    using train = def<leaky_relu, bn_con>::net_type;
    using infer = def<leaky_relu, affine>::net_type;

    // clang-format on
}  // namespace rgpnet

#endif  // RGPNet_H
