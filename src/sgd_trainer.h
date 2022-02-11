#ifndef trainer_h_INCLUDED
#define trainer_h_INCLUDED

#include "model.h"
#include "model_impl.h"

class sgd_trainer
{
    public:
    sgd_trainer() = delete;
    ~sgd_trainer();
    sgd_trainer(
        model& net,
        const float weight_decay,
        const float momentum,
        const std::vector<int>& gpus = {0});

    // load the netwok in inference mode from the synchronization file
    sgd_trainer(model& net);

    void be_verbose();
    void be_quiet();
    void print(std::ostream& out);

    void set_mini_batch_size(const size_t batch_size);
    size_t get_mini_batch_size() const;

    void set_synchronization_file(const std::string& filename);
    void load_from_synchronization_file(const std::string& filename);

    void set_learning_rate(const double lr);
    double get_learning_rate() const;

    void set_min_learning_rate(const double lr);
    double get_min_learning_rate() const;

    void set_iterations_without_progress_threshold(const size_t threshold);
    void set_test_iterations_without_progress_threshold(const size_t threshold);

    void set_learning_rate_shrink_factor(const double shrink);

    void set_learning_rate_schedule(const dlib::matrix<double>& schedule);

    size_t get_train_one_step_calls() const;

    void get_net(dlib::force_flush_to_disk force = dlib::force_flush_to_disk::yes);

    void train_one_step(
        const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
        const std::vector<std::vector<dlib::yolo_rect>>& bboxes);

    void test_one_step(
        const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
        const std::vector<std::vector<dlib::yolo_rect>>& bboxes);

    private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

#endif  // trainer_h_INCLUDED
