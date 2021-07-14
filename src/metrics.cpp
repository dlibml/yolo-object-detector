#include "model.h"
#include "utils.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>

using rgb_image = dlib::matrix<dlib::rgb_pixel>;

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
    rgb_image image;
    dlib::image_dataset_metadata::image info;
    dlib::rectangle_transform tform;
};

auto main(const int argc, const char** argv) -> int
try
{
    dlib::command_line_parser parser;
    parser.add_option("batch-size", "batch size for inference (default: 32)", 1);
    parser.add_option("conf", "detection confidence threshold (default: 0.25)", 1);
    parser.add_option("dnn", "load this network file", 1);
    parser.add_option("nms", "IoU and area covered ratio thresholds (default: 0.45 1)", 2);
    parser.add_option("nms-agnostic", "class-agnositc NMS");
    parser.add_option("size", "image size for inference (default: 512)", 1);
    parser.add_option("sync", "load this sync file", 1);
    parser.add_option("workers", "number of data loaders (default: 4)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("print", "print the network architecture");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [OPTION]... PATH/TO/DATASET/FILE.xml" << std::endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }
    parser.check_incompatible_options("dnn", "sync");
    parser.check_option_arg_range<size_t>("size", 224, 2048);
    parser.check_option_arg_range<double>("nms", 0, 1);

    dlib::file dataset_file(parser[0]);
    const auto dataset_dir = dlib::get_parent_directory(dataset_file).full_name();

    const size_t batch_size = dlib::get_option(parser, "batch-size", 32);
    const size_t image_size = dlib::get_option(parser, "size", 512);
    const size_t num_workers = dlib::get_option(parser, "workers", 512);
    const double conf_thresh = dlib::get_option(parser, "conf", 0.25);
    const std::string dnn_path = dlib::get_option(parser, "dnn", "");
    const std::string sync_path = dlib::get_option(parser, "sync", "");
    const bool classwise_nms = not parser.option("nms-agnostic");
    double iou_threshold = 0.45;
    double ratio_covered = 1.0;
    if (parser.option("nms"))
    {
        iou_threshold = std::stod(parser.option("nms").argument(0));
        ratio_covered = std::stod(parser.option("nms").argument(1));
    }

#if 1
    model_infer model;
    auto& net = model.net;

    if (not dnn_path.empty())
    {
        dlib::deserialize(dnn_path) >> net;
    }
    else if (not sync_path.empty() and dlib::file_exists(sync_path))
    {
        auto trainer = model.get_trainer();
        trainer.set_synchronization_file(sync_path);
        trainer.get_net();
    }
    else
    {
        std::cout << "ERROR: could not load the network." << std::endl;
        return EXIT_FAILURE;
    }

    net.loss_details().adjust_nms(iou_threshold, ratio_covered, classwise_nms);
    if (parser.option("print"))
        std::cout << net << std::endl;
    else
        std::cout << net.loss_details() << std::endl;
#endif

    dlib::image_dataset_metadata::dataset dataset;
    dlib::image_dataset_metadata::load_image_dataset_metadata(dataset, dataset_file.full_name());
    dlib::pipe<image_info> data(1000);
    const auto data_loader = [&](const size_t num_workers)
    {
        dlib::parallel_for(
            num_workers,
            0,
            dataset.images.size(),
            [&](size_t i)
            {
                rgb_image image;
                image_info temp;
                dlib::load_image(image, dataset_dir + "/" + dataset.images[i].filename);
                temp.info = dataset.images[i];
                temp.tform = preprocess_image(image, temp.image, image_size);
                data.enqueue(temp);
            });
    };

    // start the data loaders
    std::thread data_loaders([&]() { data_loader(num_workers); });

    std::map<std::string, result> results;
    std::map<std::string, std::vector<std::pair<double, bool>>> hits;
    std::map<std::string, unsigned long> missing;
    size_t padding = 0;
    for (const auto& label : net.loss_details().get_options().labels)
    {
        hits[label] = std::vector<std::pair<double, bool>>();
        missing[label] = 0;
        padding = std::max(label.length(), padding);
    }
    padding += 2;

    // process the dataset
    size_t num_processed = 0;
    const size_t offset = dataset.images.size() % batch_size;
    dlib::console_progress_indicator progress(dataset.images.size());
    while (num_processed != dataset.images.size())
    {
        image_info temp;
        std::vector<rgb_image> images;
        std::vector<image_info> details;
        while (images.size() < batch_size)
        {
            if (images.size() == offset and num_processed == dataset.images.size() - offset)
                break;
            data.dequeue(temp);
            images.push_back(std::move(temp.image));
            details.push_back(std::move(temp));
        }
        auto detections_batch = net.process_batch(images, batch_size, 0.005);

        for (size_t i = 0; i < images.size(); ++i)
        {
            postprocess_detections(details[i].tform, detections_batch[i]);
            const auto& im = details[i].info;
            auto& dets = detections_batch[i];
            std::vector<bool> used(dets.size(), false);
            const size_t num_pr = std::count_if(
                dets.begin(),
                dets.end(),
                [conf_thresh](const auto& d) { return d.detection_confidence > conf_thresh; });
            // true positives: truths matched by detections
            for (size_t t = 0; t < im.boxes.size(); ++t)
            {
                bool found_match_ap = false;
                bool found_match_pr = false;
                for (size_t d = 0; d < dets.size(); ++d)
                {
                    if (used[d])
                        continue;
                    if (im.boxes[t].label == dets[d].label &&
                        box_intersection_over_union(
                            dlib::drectangle(im.boxes[t].rect),
                            dets[d].rect) > 0.5)
                    {
                        used[d] = true;
                        found_match_ap = true;
                        hits.at(dets[d].label).emplace_back(dets[d].detection_confidence, true);
                        if (d < num_pr)
                        {
                            found_match_pr = true;
                            results[dets[d].label].tp++;
                        }
                        break;
                    }
                }
                // false negatives: truths not matched
                if (!found_match_ap)
                    missing.at(im.boxes[t].label)++;
                if (!found_match_pr)
                    results[im.boxes[t].label].fn++;
            }
            // false positives: detections not matched
            for (size_t d = 0; d < dets.size(); ++d)
            {
                if (!used[d])
                {
                    hits.at(dets[d].label).emplace_back(dets[d].detection_confidence, false);
                    if (d < num_pr)
                        results[dets[d].label].fp++;
                }
            }
        }
        num_processed += images.size();
        const auto percent = num_processed * 100. / dataset.images.size();
        progress.print_status(num_processed, false, std::cerr);
        std::cerr << "\t\t\t\tProgress: " << num_processed << "/" << dataset.images.size() << " ("
                  << std::fixed << std::setprecision(3) << percent << "%)        \r" << std::flush;
    }
    std::cout << std::endl;

    data.disable();
    data_loaders.join();

    double map = 0;
    double macro_precision = 0;
    double macro_recall = 0;
    double macro_f1_score = 0;
    double weighted_precision = 0;
    double weighted_recall = 0;
    double weighted_f1_score = 0;
    result micro;
    for (auto& item : hits)
    {
        std::sort(item.second.rbegin(), item.second.rend());
        const double ap = dlib::average_precision(item.second, missing[item.first]);
        const auto& r = results.at(item.first);
        micro.tp += r.tp;
        micro.fp += r.fp;
        micro.fn += r.fn;
        macro_precision += r.precision();
        macro_recall += r.recall();
        macro_f1_score += r.f1_score();
        weighted_precision += r.precision() * r.support();
        weighted_recall += r.recall() * r.support();
        weighted_f1_score += r.f1_score() * r.support();
        // clang-format off
        std::cout << dlib::rpad(item.first + ": ", padding)
                  << std::setprecision(3) << std::right << std::fixed
                  << std::setw(12) << ap * 100. << "%"
                  << std::setw(12) << r.precision()
                  << std::setw(12) << r.recall()
                  << std::setw(12) << r.f1_score()
                  << std::defaultfloat << std::setprecision(9)
                  << std::setw(12) << r.tp
                  << std::setw(12) << r.fp
                  << std::setw(12) << r.fn
                  << std::setw(12) << r.support()
                  << std::endl;
        // clang-format on
        map += ap;
    }
    size_t num_boxes = 0;
    for (const auto& im : dataset.images)
        num_boxes += im.boxes.size();

    std::cout << "--" << std::endl;
    // clang-format off
    std::cout << dlib::rpad(std::string("macro: "), padding)
              << std::setprecision(3) << std::right << std::fixed
              << std::setw(12) << map * 100. / hits.size() << "%"
              << std::setw(12) << macro_precision / results.size()
              << std::setw(12) << macro_recall / results.size()
              << std::setw(12) << macro_f1_score / results.size()
              << std::endl;
    std::cout << dlib::rpad(std::string("micro: "), padding)
              << "             "
              << std::setw(12) << micro.precision()
              << std::setw(12) << micro.recall()
              << std::setw(12) << micro.f1_score()
              << std::endl;
    std::cout << dlib::rpad(std::string("weighted: "), padding)
              << std::setprecision(3) << std::right << std::fixed
              << "             "
              << std::setw(12) << weighted_precision / num_boxes
              << std::setw(12) << weighted_recall / num_boxes
              << std::setw(12) << weighted_f1_score / num_boxes
              << std::endl;
    // clang-format on

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
