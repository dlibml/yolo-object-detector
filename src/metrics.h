#ifndef metrics_h_INCLUDED
#define metrics_h_INCLUDED

#include "utils.h"

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
        const std::string& dataset_path,
        long image_size,
        dlib::pipe<image_info>& data,
        size_t num_workers = std::thread::hardware_concurrency());

    void run();

    private:
    dlib::image_dataset_metadata::dataset dataset;
    std::string dataset_dir;
    long image_size;
    dlib::pipe<image_info>& data;
    size_t num_workers;
};

std::pair<double, double> compute_map(
    net_infer_type& net,
    const dlib::image_dataset_metadata::dataset& dataset,
    const size_t batch_size,
    dlib::pipe<image_info>& data,
    double conf_thresh = 0.25,
    std::ostream& out = std::cout
);

void save_model(
    net_infer_type& net,
    const std::string& sync_path,
    long num_steps,
    double map,
    double wf1);

#endif  // metrics_h_INCLUDED
