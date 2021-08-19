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
    template <template <typename> class ACT, template <typename> class BN, long k = 32>
    struct def
    {

        template <long num_filters, long ks, int s, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, ks/2, ks/2>, SUBNET>;

        template <long nf, int ks, int s, typename SUBNET>
        using conblock = ACT<BN<conp<nf, ks, s, SUBNET>>>;

        template <typename INPUT>
        using stem = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, ACT<BN<conp<2 * k, 7, 2, INPUT>>>>;

        template <long num_filters, typename SUBNET>
        using transition = avg_pool<2, 2, 2, 2, con<num_filters, 1, 1, 1, 1, ACT<BN<SUBNET>>>>;

        template <typename SUBNET>
        using dense_layer = concat2<tag1, tag2,
                            tag2<conp<k, 3, 1,
                            ACT<BN<conp<4 * k, 1, 1,
                            ACT<BN<tag1<SUBNET>>>>>>>>>;

        template <size_t n4, size_t n3, size_t n2, size_t n1, typename INPUT>
        using densenet =
            btag4<ACT<BN<repeat<n4, dense_layer, transition<k * (2 + n1 + 2 * n2 + 4 * n3) / 8,
                   btag3<repeat<n3, dense_layer, transition<k * (2 + n1 + 2 * n2) / 4,
                   btag2<repeat<n2, dense_layer, transition<k * (2 + n1) / 2,
                   btag1<repeat<n1, dense_layer, stem<INPUT>>>>>>>>>>>>>>;

        // the darknet block
        template<long num_filters, typename SUBNET>
        using darkblock = conblock<num_filters, 3, 1,
                conblock<num_filters / 2, 1, 1,
                conblock<num_filters, 3, 1,
                conblock<num_filters, 1, 1, SUBNET>>>>;

        // --------------------------------- RGPNet --------------------------------- //

        template <long num_filters, typename SUBNET>
        using downsampler = add_layer<con_<num_filters, 3, 3, 2, 2, 1, 1>, SUBNET>;
        template <long num_filters, typename SUBNET>
        using upsampler = upsample<2, con<num_filters, 1, 1, 1, 1, SUBNET>>;

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
        darkblock<feats4, concat2<itag4, atag0, atag0<in_lvl3d<itag4<in_lvl4<SUBNET>>>>>>;

        // adaptor3 adds in_lvl3, inlvl2d and in_lvl4u
        template <typename SUBNET> using adaptor3 =
        darkblock<feats3, concat3<itag3, atag0, atag1, atag1<in_lvl2d<atag0<upsampler<feats3, SUBNET>>>>>>;

        // adaptor2 adds in_lvl2, in_lvl1d, and inlvl3u
        template <typename SUBNET> using adaptor2 =
        darkblock<feats2, concat3<itag2, atag0, atag1, atag1<in_lvl1d<atag0<upsampler<feats2, SUBNET>>>>>>;

        // adaptor1 adds in_lvl1 and inlvl2u
        template <typename SUBNET> using adaptor1 =
        darkblock<feats1, concat2<itag1, atag0, atag0<upsampler<feats1, SUBNET>>>>;

        template <typename SUBNET>
        using spp = concat4<tag4, tag3, tag2, tag1,
               tag4<max_pool<13, 13, 1, 1,
                    skip1<
               tag3<max_pool<9, 9, 1, 1,
                    skip1<
               tag2<max_pool<5, 5, 1, 1,
               tag1<SUBNET>>>>>>>>>>;

        template <typename INPUT>
        using backbone = densenet<16, 24, 12, 6, INPUT>;

        using net_type = loss_yolo<ytag8, ytag16, ytag32,
        ytag8<sig<con<1, 1, 1, 1, 1, adaptor2<askip3<
        ytag16<sig<con<1, 1, 1, 1, 1, atag3<adaptor3<askip4<
        ytag32<sig<con<1, 1, 1, 1, 1, atag4<adaptor4<
        spp<backbone<input_rgb_image>>>>>>>>>>>>>>>>>>>;
    };

    using train = def<leaky_relu, bn_con, 16>::net_type;
    using infer = def<leaky_relu, affine, 16>::net_type;

    // clang-format on
}  // namespace rgpnet

#endif  // RGPNet_H
