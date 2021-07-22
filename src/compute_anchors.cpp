#include <dlib/clustering.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/svm.h>

using sample_t = dlib::matrix<double, 2, 1>;

auto main(const int argc, const char** argv) -> int
try
{
    dlib::command_line_parser parser;
    parser.add_option("dataset", "path to the dataset XML file", 1);
    parser.add_option("size", "image size to use during training (default: 512)", 1);
    parser.add_option("sides", "min and max sides covered for an anchor box group", 2);
    parser.add_option("clusters", "number of clusters for an anchor box group", 1);
    parser.add_option("iou", "minimum IoU each anchor should have", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        std::cout << "Notes:\n";
        std::cout << dlib::wrap_string(
                         "1: --clusters and --iou are incompatible, because --iou "
                         "will find the number of clusters automatically.",
                         0,
                         3)
                  << std::endl;
        ;
        std::cout << dlib::wrap_string(
                         "2: --sides and --clusters should be specified as many "
                         "times as strides, and they must match.",
                         0,
                         3)
                  << std::endl;
        return EXIT_SUCCESS;
    }

    parser.check_incompatible_options("clusters", "iou");
    parser.check_option_arg_range("iou", 0.0, 1.0);

    const size_t image_size = dlib::get_option(parser, "size", 512);
    const double min_iou = dlib::get_option(parser, "iou", 0.5);
    const std::string dataset_path = dlib::get_option(parser, "dataset", "");
    if (dataset_path.empty())
    {
        std::cout << "specify the data path directory" << std::endl;
        return EXIT_FAILURE;
    }

    if (not parser.option("sides"))
    {
        std::cout << "specify the sides of an anchor box group." << std::endl;
        return EXIT_FAILURE;
    }
    const size_t num_groups = parser.option("sides").count();

    if (not parser.option("clusters") and not parser.option("iou"))
    {
        std::cout << "specify the number of clusters for an anchor box group or the minimum IoU."
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (parser.option("clusters"))
    {
        DLIB_CASSERT(parser.option("sides").count() == parser.option("clusters").count());
    }

    // Load the dataset
    dlib::image_dataset_metadata::dataset dataset;
    dlib::image_dataset_metadata::load_image_dataset_metadata(dataset, dataset_path);

    // Prepare the anchor box groups
    std::vector<size_t> clusters;
    std::vector<std::pair<double, double>> ranges;
    for (size_t i = 0; i < parser.option("sides").count(); ++i)
    {
        std::pair<double, double> range;
        range.first = std::stod(parser.option("sides").argument(0, i));
        range.second = std::stod(parser.option("sides").argument(1, i));
        if (range.first > range.second)
            std::swap(range.first, range.second);
        ranges.push_back(std::move(range));
        if (parser.option("clusters"))
            clusters.push_back(std::stoul(parser.option("clusters").argument(0, i)));

        std::cout << "group #" << i << ": " << ranges.back().first << " - " << ranges.back().second
                  << std::endl;
    }

    // Group the ground truth boxes by area covered
    size_t num_boxes = 0;
    std::vector<std::vector<sample_t>> box_groups(num_groups);
    for (const auto& image_info : dataset.images)
    {
        const auto scale = image_size / std::max<double>(image_info.width, image_info.height);
        for (const auto& box : image_info.boxes)
        {
            sample_t sample;
            sample(0) = box.rect.width() * scale;
            sample(1) = box.rect.height() * scale;
            const auto [min_side, max_side] = std::minmax(sample(0), sample(1));
            for (size_t i = 0; i < ranges.size(); ++i)
            {
                if (ranges[i].second > max_side)
                {
                    box_groups.at(i).push_back(std::move(sample));
                    break;
                }
            }
            ++num_boxes;
        }
    }

    const auto compute_average_iou =
        [](const std::vector<sample_t>& samples, const std::vector<sample_t>& anchors)
    {
        double average_iou = 0;
        for (const auto& s : samples)
        {
            const auto sample = dlib::centered_drect(dlib::dpoint(0, 0), s(0), s(1));
            double best_iou = 0;
            for (const auto& a : anchors)
            {
                const auto anchor = centered_drect(dlib::dpoint(0, 0), a(0), a(1));
                best_iou = std::max(best_iou, box_intersection_over_union(sample, anchor));
            }
            average_iou += best_iou;
        }
        return average_iou / samples.size();
    };

    std::cout << "total number of boxes: " << num_boxes << std::endl;
    if (parser.option("clusters"))
    {
        for (size_t i = 0; i < num_groups; ++i)
        {
            auto& samples = box_groups[i];
            const auto num_clusters = clusters[i];
            dlib::randomize_samples(samples);
            std::cout << "Computing anchors for " << samples.size() << " samples" << std::endl;
            std::vector<sample_t> anchors;
            dlib::pick_initial_centers(num_clusters, anchors, samples);
            dlib::find_clusters_using_kmeans(samples, anchors);
            std::sort(
                anchors.begin(),
                anchors.end(),
                [](const auto& a, const auto& b) { return a(0) * a(1) < b(0) * b(1); });
            for (const auto& c : anchors)
                std::cout << "    " << round(c(0)) << 'x' << round(c(1)) << std::endl;
            // And check the average IoU of the newly computed anchor boxes and the training
            // samples.
            std::cout << "  Average IoU: " << compute_average_iou(samples, anchors) << std::endl;
        }
    }
    else if (parser.option("iou"))
    {
        dlib::rand rnd;
        dlib::test_box_overlap overlaps(min_iou);

        const auto count_overlaps = [](const std::vector<sample_t>& samples,
                                       const dlib::test_box_overlap& overlaps,
                                       const sample_t& ref_sample)
        {
            const auto ref_box =
                dlib::centered_drect(dlib::dpoint(0, 0), ref_sample(0), ref_sample(1));
            size_t cnt = 0;
            for (const auto& s : samples)
            {
                const auto b = dlib::centered_drect(dlib::dpoint(0, 0), s(0), s(1));
                if (overlaps(b, ref_box))
                    ++cnt;
            }
            return cnt;
        };

        const auto find_samples_overlapping_all_others =
            [count_overlaps](std::vector<sample_t> samples, const dlib::test_box_overlap overlaps)
        {
            std::vector<sample_t> exemplars;
            dlib::rand rnd;
            while (samples.size() > 0)
            {
                sample_t best_ref_sample;
                best_ref_sample = 0, 0;
                dlib::randomize_samples(samples);
                size_t best_cnt = 0;
                for (size_t i = 0; i < 500; ++i)
                {
                    const auto sample = samples[rnd.get_random_64bit_number() % samples.size()];
                    const auto cnt = count_overlaps(samples, overlaps, sample);
                    if (cnt > best_cnt)
                    {
                        best_cnt = cnt;
                        best_ref_sample = sample;
                    }
                }

                const auto best_ref_box = dlib::centered_drect(
                    dlib::dpoint(0, 0),
                    best_ref_sample(0),
                    best_ref_sample(1));
                for (size_t i = 0; i < samples.size(); ++i)
                {
                    const auto b =
                        dlib::centered_drect(dlib::dpoint(0, 0), samples[i](0), samples[i](1));
                    if (overlaps(b, best_ref_box))
                    {
                        std::swap(samples[i], samples.back());
                        samples.pop_back();
                        --i;
                    }
                }
                exemplars.push_back(best_ref_sample);
            }
            return exemplars;
        };

        for (size_t g = 0; g < num_groups; ++g)
        {
            auto samples = box_groups[g];
            const auto anchors = find_samples_overlapping_all_others(samples, overlaps);
            std::cout << "# anchors: " << anchors.size() << std::endl;
            for (const auto& c : anchors)
                std::cout << "    " << round(c(0)) << 'x' << round(c(1)) << std::endl;
            std::cout << "  Average IoU: " << compute_average_iou(samples, anchors) << std::endl;
        }
    }
    else
    {
        std::cout << "Nothing happened." << std::endl;
    }
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
