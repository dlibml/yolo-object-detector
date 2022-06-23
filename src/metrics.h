#ifndef metrics_h_INCLUDED
#define metrics_h_INCLUDED

#include "model.h"

#include <dlib/data_io.h>
#include <dlib/pipe.h>

struct result
{
    result() = default;
    double tp = 0;
    double fp = 0;
    double fn = 0;
    double precision() const { return retrieved() == 0 ? 0 : tp / retrieved(); }
    double recall() const { return relevant() == 0 ? 0 : tp / relevant(); }
    double f1_score() const { return pr() == 0 ? 0 : 2.0 * precision() * recall() / pr(); }
    double support() const { return relevant(); }

    private:
    double retrieved() const { return tp + fp; }
    double relevant() const { return tp + fn; }
    double pr() const { return precision() + recall(); }
};

struct image_info
{
    dlib::matrix<dlib::rgb_pixel> image;
    dlib::image_dataset_metadata::image info;
    dlib::rectangle_transform tform;
};

class test_data_loader
{
    public:
    test_data_loader() = delete;
    test_data_loader(
        const std::string& dataset_dir,
        const dlib::image_dataset_metadata::dataset& dataset,
        dlib::pipe<image_info>& data,
        long image_size = 512,
        size_t num_workers = std::thread::hardware_concurrency());

    void run();

    private:
    const std::string& dataset_dir;
    const dlib::image_dataset_metadata::dataset& dataset;
    dlib::pipe<image_info>& data;
    long image_size;
    size_t num_workers;
};

struct metrics_details
{
    double map = 0;
    double macro_p = 0;
    double macro_r = 0;
    double macro_f = 0;
    double micro_p = 0;
    double micro_r = 0;
    double micro_f = 0;
    double weighted_p = 0;
    double weighted_r = 0;
    double weighted_f = 0;
};

inline std::ostream& operator<<(std::ostream& out, const metrics_details& item)
{
    out << item.map << ' ' << item.macro_p << ' ' << item.macro_r << ' ' << item.macro_f << ' '
        << item.micro_p << ' ' << item.micro_r << ' ' << item.micro_f << ' ' << item.weighted_p
        << ' ' << item.weighted_r << ' ' << item.weighted_f;
    return out;
}

metrics_details compute_metrics(
    model& net,
    const dlib::image_dataset_metadata::dataset& dataset,
    const size_t batch_size,
    dlib::pipe<image_info>& data,
    const double conf_thresh = 0.25,
    std::ostream& out = std::cout);

void save_model(
    model& net,
    const std::string& name,
    size_t num_steps,
    double map,
    double wf1);

#endif  // metrics_h_INCLUDED
