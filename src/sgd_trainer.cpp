#include "sgd_trainer.h"

using namespace dlib;
struct sgd_trainer::impl
{
    impl() = delete;
    ~impl() = default;
    impl(model& net, const float weight_decay, const float momentum, const std::vector<int> gpus)
        : trainer(std::make_unique<dnn_trainer<net_train_type, sgd>>(
              net.pimpl->train,
              sgd(weight_decay, momentum),
              gpus)),
          loader(std::make_unique<dnn_trainer<net_infer_type, sgd>>(net.pimpl->infer))
    {
    }

    impl(model& net)
        : trainer(std::make_unique<dnn_trainer<net_train_type, sgd>>(net.pimpl->train)),
          loader(std::make_unique<dnn_trainer<net_infer_type, sgd>>(net.pimpl->infer))
    {
    }

    std::unique_ptr<dnn_trainer<net_train_type, sgd>> trainer;
    std::unique_ptr<dnn_trainer<net_infer_type, sgd>> loader;
};

sgd_trainer::~sgd_trainer() = default;

sgd_trainer::sgd_trainer(
    model& net,
    const float weight_decay,
    const float momentum,
    const std::vector<int>& gpus)
    : pimpl(std::make_unique<sgd_trainer::impl>(net, weight_decay, momentum, gpus))
{
}

sgd_trainer::sgd_trainer(model& net) : pimpl(std::make_unique<sgd_trainer::impl>(net))
{
}

void sgd_trainer::be_verbose()
{
    pimpl->trainer->be_verbose();
}

void sgd_trainer::be_quiet()
{
    pimpl->trainer->be_quiet();
}

void sgd_trainer::print(std::ostream& out)
{
    out << *(pimpl->trainer.get()) << '\n';
}

void sgd_trainer::set_mini_batch_size(const size_t batch_size)
{
    pimpl->trainer->set_mini_batch_size(batch_size);
}

size_t sgd_trainer::get_mini_batch_size() const
{
    return pimpl->trainer->get_mini_batch_size();
}

void sgd_trainer::set_synchronization_file(const std::string& filename)
{
    pimpl->trainer->set_synchronization_file(filename, std::chrono::minutes(30));
}

void sgd_trainer::load_from_synchronization_file(const std::string& filename)
{
    pimpl->loader->set_synchronization_file(filename);
    pimpl->loader->get_net();
}

void sgd_trainer::set_learning_rate(const double lr)
{
    pimpl->trainer->set_learning_rate(lr);
}

double sgd_trainer::get_learning_rate() const
{
    return pimpl->trainer->get_learning_rate();
}

void sgd_trainer::set_min_learning_rate(const double lr)
{
    pimpl->trainer->set_min_learning_rate(lr);
}

double sgd_trainer::get_min_learning_rate() const
{
    return pimpl->trainer->get_min_learning_rate();
}

void sgd_trainer::set_iterations_without_progress_threshold(const size_t threshold)
{
    pimpl->trainer->set_iterations_without_progress_threshold(threshold);
}

void sgd_trainer::set_test_iterations_without_progress_threshold(const size_t threshold)
{
    pimpl->trainer->set_test_iterations_without_progress_threshold(threshold);
}

void sgd_trainer::set_learning_rate_shrink_factor(const double shrink)
{
    pimpl->trainer->set_learning_rate_shrink_factor(shrink);
}

void sgd_trainer::set_learning_rate_schedule(const dlib::matrix<double>& schedule)
{
    pimpl->trainer->set_learning_rate_schedule(schedule);
}

size_t sgd_trainer::get_train_one_step_calls() const
{
    return pimpl->trainer->get_train_one_step_calls();
}

void sgd_trainer::get_net(dlib::force_flush_to_disk force)
{
    pimpl->trainer->get_net(force);
}

void sgd_trainer::train_one_step(
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
    const std::vector<std::vector<dlib::yolo_rect>>& bboxes)
{
    pimpl->trainer->train_one_step(images, bboxes);
}

void sgd_trainer::test_one_step(
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
    const std::vector<std::vector<dlib::yolo_rect>>& bboxes)
{
    pimpl->trainer->test_one_step(images, bboxes);
}
