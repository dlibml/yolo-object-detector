#ifndef yolov7_tiny_h_INCLUDED
#define yolov7_tiny_h_INCLUDED

#include <dlib/dnn.h>

namespace yolov7_tiny
{
    using namespace dlib;
    template <typename SUBNET> using ytag3 = add_tag_layer<4003, SUBNET>;
    template <typename SUBNET> using ytag4 = add_tag_layer<4004, SUBNET>;
    template <typename SUBNET> using ytag5 = add_tag_layer<4005, SUBNET>;
    template <typename SUBNET> using ptag3 = add_tag_layer<7003, SUBNET>;
    template <typename SUBNET> using ptag4 = add_tag_layer<7004, SUBNET>;
    template <typename SUBNET> using ptag5 = add_tag_layer<7005, SUBNET>;
    template <typename SUBNET> using ntag4 = add_tag_layer<5004, SUBNET>;
    template <typename SUBNET> using ntag5 = add_tag_layer<5005, SUBNET>;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {

        template <long NF, int KS, int S, typename SUBNET>
        using conv = ACT<BN<add_layer<con_<NF, KS, KS, S, S, (KS-1)/2, (KS-1)/2>, SUBNET>>>;

        template <long NF, typename SUBNET>
        using e_elan = conv<NF, 1, 1,
                       concat4<itag4, itag3, itag2, itag1,
                 itag4<conv<NF / 2, 3, 1,
                 itag3<conv<NF / 2, 3, 1,
                 itag2<conv<NF / 2, 1, 1, iskip<
                 itag1<conv<NF / 2, 1, 1,
                 itag0<SUBNET>>>>>>>>>>>>;

        template <typename INPUT>
        using backbone = ptag5<e_elan<512, max_pool<2, 2, 2, 2,
                         ptag4<e_elan<256, max_pool<2, 2, 2, 2,
                         ptag3<e_elan<128, max_pool<2, 2, 2, 2,
                               e_elan<64, conv<64, 3, 2, conv<32, 3, 2,
                               INPUT>>>>>>>>>>>>;

        template <long NF, typename SUBNET>
        using csp_spp = conv<NF, 1, 1,
                        concat2<itag5, tag10,
                  itag5<conv<NF, 1, 1,
                        concat4<itag1, itag2, itag3, itag4,
                  itag4<max_pool<5, 5, 1, 1,
                  itag3<max_pool<5, 5, 1, 1,
                  itag2<max_pool<5, 5, 1, 1,
                  itag1<conv<NF, 1, 1, iskip<
                  tag10<conv<NF, 1, 1,
                  itag0<SUBNET>>>>>>>>>>>>>>>>>;

        template <template <typename> class YTAG, typename SUBNET>
        using yolo = YTAG<sig<con<255, 1, 1, 1, 1, SUBNET>>>;

        template <typename SUBNET>
        using head = yolo<ytag5, conv<512, 3, 1,
                     e_elan<256,
                     concat2<tag5, ntag5,
                tag5<conv<256, 3, 2, skip1<
                     yolo<ytag4, conv<256, 3, 1,
                tag1<e_elan<128,
                     concat2<tag4, ntag4,
                tag4<conv<128, 3, 2, skip1<
                     yolo<ytag3, conv<128, 3, 1,
                tag1<e_elan<64,
                     concat2<tag2, tag1,
                tag2<conv<64, 1, 1, add_skip_layer<ptag3,
                tag1<upsample<2,
                     conv<64, 1, 1,
               ntag4<e_elan<128,
                     concat2<tag2, tag1,
                tag2<conv<128, 1, 1, add_skip_layer<ptag4,
                tag1<upsample<2,
                     conv<128, 1, 1,
               ntag5<csp_spp<256,
                     SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

        using net_type = loss_yolo<ytag3, ytag4, ytag5, head<backbone<input_rgb_image>>>;
    };

    using train_type = def<leaky_relu, bn_con>::net_type;
    using infer_type = def<leaky_relu, affine>::net_type;
}

#endif // yolov7_tiny_h_INCLUDED
