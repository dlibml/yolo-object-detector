#include "metrics.h"

#include "detector_utils.h"

test_data_loader::test_data_loader(
    const std::string& dataset_dir,
    const dlib::image_dataset_metadata::dataset& dataset,
    dlib::pipe<image_info>& data,
    long image_size,
    size_t num_workers)
    : dataset_dir(dataset_dir),
      dataset(dataset),
      data(data),
      image_size(image_size),
      num_workers(num_workers)
{
}

void test_data_loader::run()
{
    dlib::parallel_for(
        num_workers,
        0,
        dataset.images.size(),
        [&](size_t i)
        {
            dlib::matrix<dlib::rgb_pixel> image;
            image_info temp;
            dlib::load_image(image, dataset_dir + "/" + dataset.images[i].filename);
            temp.info = dataset.images[i];
            temp.tform = preprocess_image(image, temp.image, image_size);
            data.enqueue(temp);
        });
}

metrics_details compute_metrics(
    model& net,
    const dlib::image_dataset_metadata::dataset& dataset,
    const size_t batch_size,
    dlib::pipe<image_info>& data,
    double conf_thresh,
    std::ostream& out)
{
    std::map<std::string, result> results;
    std::map<std::string, std::vector<std::pair<double, bool>>> hits;
    std::map<std::string, unsigned long> missing;
    size_t padding = 0;
    for (const auto& label : net.get_options().labels)
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
        std::vector<dlib::matrix<dlib::rgb_pixel>> images;
        std::vector<image_info> details;
        while (images.size() < batch_size)
        {
            if (images.size() == offset and num_processed == dataset.images.size() - offset)
                break;
            data.dequeue(temp);
            images.push_back(std::move(temp.image));
            details.push_back(std::move(temp));
        }
        auto detections_batch = net(images, batch_size, 0.001);

        for (size_t i = 0; i < images.size(); ++i)
        {
            const auto& im = details[i].info;
            auto& dets = detections_batch[i];
            postprocess_detections(details[i].tform, dets);
            std::vector<bool> used(dets.size(), false);
            const size_t num_pr = std::count_if(
                dets.begin(),
                dets.end(),
                [conf_thresh](const auto& d) { return d.detection_confidence >= conf_thresh; });
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
        progress.print_status(num_processed, false, std::clog);
    }
    out << std::endl;

    metrics_details metrics;
    result micro;
    for (auto& item : hits)
    {
        std::sort(item.second.rbegin(), item.second.rend());
        const double ap = dlib::average_precision(item.second, missing[item.first]);
        const auto& r = results.at(item.first);
        micro.tp += r.tp;
        micro.fp += r.fp;
        micro.fn += r.fn;
        metrics.macro_p += r.precision();
        metrics.macro_r += r.recall();
        metrics.macro_f += r.f1_score();
        metrics.weighted_p += r.precision() * r.support();
        metrics.weighted_r += r.recall() * r.support();
        metrics.weighted_f += r.f1_score() * r.support();
        // clang-format off
        out << dlib::rpad(item.first + ": ", padding)
                  << std::setprecision(4) << std::right << std::fixed
                  << std::setw(12) << ap
                  << std::setprecision(4)
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
        metrics.map += ap;
    }
    size_t num_boxes = 0;
    for (const auto& im : dataset.images)
        num_boxes += im.boxes.size();

    metrics.map /= hits.size();
    metrics.macro_p /= results.size();
    metrics.macro_r /= results.size();
    metrics.macro_f /= results.size();
    metrics.micro_p = micro.precision();
    metrics.micro_r = micro.recall();
    metrics.micro_f = micro.f1_score();
    metrics.weighted_p /= num_boxes;
    metrics.weighted_r /= num_boxes;
    metrics.weighted_f /= num_boxes;
    out << "--" << std::endl;
    // clang-format off
    out << dlib::rpad(std::string("macro: "), padding)
              << std::setprecision(4) << std::right << std::fixed
              << std::setw(12) << metrics.map
              << std::setw(12) << metrics.macro_p
              << std::setw(12) << metrics.macro_r
              << std::setw(12) << metrics.macro_f
              << std::endl;
    out << dlib::rpad(std::string("micro: "), padding)
              << "            "
              << std::setw(12) << metrics.micro_p
              << std::setw(12) << metrics.micro_r
              << std::setw(12) << metrics.micro_f
              << std::endl;
    out << dlib::rpad(std::string("weighted: "), padding)
              << std::setprecision(4) << std::right << std::fixed
              << "            "
              << std::setw(12) << metrics.weighted_p
              << std::setw(12) << metrics.weighted_r
              << std::setw(12) << metrics.weighted_f
              << std::endl;
    // clang-format on
    return metrics;
}

void save_model(model& net, const std::string& name, size_t num_steps, double map, double wf1)
{
    std::stringstream filename;
    filename << name << "_step-" << dlib::pad_int_with_zeros(num_steps);
    filename << "_map-" << std::fixed << std::setprecision(4) << map;
    filename << "_wf1-" << std::fixed << std::setprecision(4) << wf1;
    filename << ".dnn";
    net.save_train(filename.str());
    std::cout << "model saved as: " << filename.str() << '\n';
}
