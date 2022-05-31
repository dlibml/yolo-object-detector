#ifndef model_h_INCLUDED
#define model_h_INCLUDED

#include <dlib/dnn.h>

template <typename SUBNET> using ytag3 = dlib::add_tag_layer<4003, SUBNET>;
template <typename SUBNET> using ytag4 = dlib::add_tag_layer<4004, SUBNET>;
template <typename SUBNET> using ytag5 = dlib::add_tag_layer<4005, SUBNET>;

class model
{
    public:
    model();
    ~model();
    model(const dlib::yolo_options& options);
    auto operator()(const dlib::matrix<dlib::rgb_pixel>& image, const float conf = 0.25)
        -> std::vector<dlib::yolo_rect>;

    auto operator()(
        const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
        const size_t batch_size,
        const float conf = 0.25) -> std::vector<std::vector<dlib::yolo_rect>>;

    void sync();
    void clean();
    void save_train(const std::string& path);
    void load_train(const std::string& path);
    void save_infer(const std::string& path);
    void load_infer(const std::string& path);
    void load_backbone(const std::string& path);
    auto get_strides(const long image_size = 512) -> std::vector<long>;
    const dlib::yolo_options& get_options() const;
    void adjust_nms(
        const float iou_threshold,
        const float ratio_covered = 1,
        const bool classwise = true);
    void fuse();
    void print(std::ostream& out) const;
    void print_loss_details(std::ostream& out = std::cout) const;

    private:
    struct impl;
    std::unique_ptr<impl> pimpl;
    friend class sgd_trainer;
};

#endif  // model_h_INCLUDED
