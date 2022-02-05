#ifndef yolor_h_INCLUDED
#define yolor_h_INCLUDED

#include <dlib/dnn.h>

namespace yolor
{
    using namespace dlib;
    template <typename SUBNET> using ytag3 = add_tag_layer<4003, SUBNET>;
    template <typename SUBNET> using ytag4 = add_tag_layer<4004, SUBNET>;
    template <typename SUBNET> using ytag5 = add_tag_layer<4005, SUBNET>;
    template <typename SUBNET> using ytag6 = add_tag_layer<4006, SUBNET>;
    template <typename SUBNET> using ptag3 = add_tag_layer<7003, SUBNET>;
    template <typename SUBNET> using ptag4 = add_tag_layer<7004, SUBNET>;
    template <typename SUBNET> using ptag5 = add_tag_layer<7005, SUBNET>;
    template <typename SUBNET> using ptag6 = add_tag_layer<7006, SUBNET>;
    // clang-format off
    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long nf, int ks, int s, typename SUBNET>
        using conv = ACT<BN<add_layer<con_<nf, ks, ks, s, s, ks/2, ks/2>, SUBNET>>>;

        template <long num_filters, typename SUBNET>
        using bottleneck = add_prev10<conv<num_filters, 3, 1, conv<num_filters, 1, 1, tag10<SUBNET>>>>;

        template <typename SUBNET> using bottleneck_64 = bottleneck<64, SUBNET>;
        template <typename SUBNET> using bottleneck_128 = bottleneck<128, SUBNET>;
        template <typename SUBNET> using bottleneck_192 = bottleneck<192, SUBNET>;
        template <typename SUBNET> using bottleneck_256 = bottleneck<256, SUBNET>;
        template <typename SUBNET> using bottleneck_320 = bottleneck<320, SUBNET>;
        template <typename SUBNET> using bottleneck_384 = bottleneck<384, SUBNET>;
        template <typename SUBNET> using bottleneck_512 = bottleneck<512, SUBNET>;
        template <typename SUBNET> using bottleneck_640 = bottleneck<640, SUBNET>;

        template <long num_filters, size_t N, template <typename> class BLOCK, typename SUBNET>
        using bottleneck_cspf = conv<num_filters, 1, 1,
                                ACT<BN<concat2<tag8, tag9,
                           tag9<con<num_filters/2, 1, 1, 1, 1, skip7<
                           tag8<repeat<N, BLOCK, conv<num_filters/2, 1, 1,
                           tag7<SUBNET>>>>>>>>>>>;

        template <long num_filters, size_t N, template <typename> class BLOCK, typename SUBNET>
        using bottleneck_csp2 = conv<num_filters, 1, 1,
                                ACT<BN<concat2<tag8, tag9,
                           tag9<con<num_filters, 1, 1, 1, 1, skip7<
                           tag8<repeat<N, BLOCK,
                           tag7<conv<num_filters, 1, 1, SUBNET>>>>>>>>>>>;

        template <typename INPUT> using backbone =
        ptag6<bottleneck_cspf<640, 3, bottleneck_320,
              conv<640, 3, 2,
        ptag5<bottleneck_cspf<512, 3, bottleneck_256,
              conv<512, 3, 2,
        ptag4<bottleneck_cspf<384, 7, bottleneck_192,
              conv<384, 3, 2,
        ptag3<bottleneck_cspf<256, 7, bottleneck_128,
              conv<256, 3, 2,
              bottleneck_cspf<128, 3, bottleneck_64,
              conv<128, 3, 2,
              conv<64, 3, 1,
              reorg<INPUT>
        >>>>>>>>>>>>>>>;

        template <long num_filters, typename SUBNET>
        using spp_csp =
                    conv<num_filters, 1, 1,
                    ACT<BN<concat2<tag5, tag6,
               tag6<con<num_filters, 1, 1, 1, 1, skip1<
               tag5<conv<num_filters, 3, 1, conv<num_filters, 1, 1,
                    concat4<tag1, tag2, tag3, tag4,
               tag4<max_pool<13, 13, 1, 1,
                    skip1<
               tag3<max_pool<9, 9, 1, 1,
                    skip1<
               tag2<max_pool<5, 5, 1, 1,
               tag1<conv<num_filters, 1, 1,
                    conv<num_filters, 3, 1,
                    conv<num_filters, 1, 1,
                    SUBNET>>>>>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class YTAG, typename SUBNET>
        using yolo = YTAG<sig<con<1, 1, 1, 1, 1, SUBNET>>>;

        template <typename SUBNET>
        using head = add_loss_layer<loss_yolo_<ytag3, ytag4, ytag5, ytag6>,
                 yolo<ytag6,
                 conv<640, 3, 1,
                 bottleneck_csp2<320, 3, bottleneck_320,
                 concat2<tag1, tag6,
            tag1<conv<320, 3, 2, skip1<
                 yolo<ytag5,
            tag1<conv<512, 3, 1,
                 bottleneck_csp2<256, 3, bottleneck_256,
                 concat2<tag1, tag5,
            tag1<conv<256, 3, 2, skip1<
                 yolo<ytag4,
            tag1<conv<384, 3, 1,
                 bottleneck_csp2<192, 3, bottleneck_192,
                 concat2<tag1, tag4,
            tag1<conv<192, 3, 2, skip1<
                 yolo<ytag3,
            tag1<conv<256, 3, 1,
                 bottleneck_csp2<128, 3, bottleneck_128,
                 concat2<tag1, tag2,
            tag2<conv<128, 1, 1, add_skip_layer<ptag3,
            tag1<upsample<2,
                 conv<128, 1, 1,
            tag4<bottleneck_csp2<192, 3, bottleneck_192,
                 concat2<tag1, tag2,
            tag2<conv<192, 1, 1, add_skip_layer<ptag4,
            tag1<upsample<2,
                 conv<192, 1, 1,
            tag5<bottleneck_csp2<256, 3, bottleneck_256,
                 concat2<tag1, tag2,
            tag2<conv<256, 1, 1, add_skip_layer<ptag5,
            tag1<upsample<2,
                 conv<256, 1, 1,
            tag6<spp_csp<320, SUBNET>>
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

        using net_type = head<backbone<input_rgb_image>>;

    };

    using train_type = def<leaky_relu, bn_con>::net_type;
    using infer_type = def<leaky_relu, affine>::net_type;
    // clang-format on
}  // namespace yolor

#endif  // yolor_h_INCLUDED
