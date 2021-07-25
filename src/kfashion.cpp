#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <nlohmann/json.hpp>
#include <regex>

using json = nlohmann::json;

std::unordered_map<std::string, std::string> translation{
    {"상의", "top"},
    {"아우터", "outer"},
    {"원피스", "wholebody"},
    {"하의", "pants"},  // I know it means bottom, but pants is more likely than skirt
};


auto main(const int argc, const char** argv) -> int
try
{
    dlib::command_line_parser parser;
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() == 0 or parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const auto data_dir = dlib::directory(parser[0]);
    const auto files = dlib::get_files_in_directory_tree(data_dir, dlib::match_ending(".json"));
    std::clog << "number of files: " << files.size() << std::endl;

    dlib::image_dataset_metadata::dataset dataset;
    dataset.name = "K-Fashion";
    dataset.comment = "validation set";
    dataset.images.resize(files.size());
    dlib::parallel_for_verbose(0, files.size(), [&](size_t i)
    {
        const auto& file = files[i];
        std::ifstream fin(file.full_name());
        json data;
        fin >> data;
        // image metadata info: filename, width and height
        auto& image = dataset.images[i];
        image.filename = data["이미지 정보"]["이미지 파일명"].get<std::string>();
        image.filename = dlib::get_parent_directory(file).full_name() + "/" + image.filename;
        image.filename.replace(image.filename.find("labels"), 6, "images");
        image.width = data["이미지 정보"]["이미지 너비"].get<unsigned long>();
        image.height = data["이미지 정보"]["이미지 높이"].get<unsigned long>();
        auto details = data["데이터셋 정보"]["데이터셋 상세설명"]["렉트좌표"];
        for (const auto& detail : details.items())
        {
            if (not detail.value()[0].empty())
            {
                for (const auto& coords : detail.value())
                {
                    // std::cout << coords << std::endl;
                    dlib::image_dataset_metadata::box box;
                    const auto x = coords["X좌표"].get<double>();
                    const auto y = coords["Y좌표"].get<double>();
                    const auto w = coords["가로"].get<double>();
                    const auto h = coords["세로"].get<double>();
                    box.rect.left() = std::round(x);
                    box.rect.top() = std::round(y);
                    box.rect.right() = std::round(x + w);
                    box.rect.bottom() = std::round(y + h);
                    box.label = translation.at(detail.key());
                    image.boxes.push_back(std::move(box));
                }
            }
        }
    });
    dlib::image_dataset_metadata::save_image_dataset_metadata(dataset, "dataset.xml");

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
