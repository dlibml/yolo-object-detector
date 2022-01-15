#include <dlib/data_io.h>
#include <nlohmann/json.hpp>

using namespace dlib;
using json = nlohmann::json;

struct category
{
    category(const std::string& name, const std::string& super) : name(name), super(super) {}
    const std::string name{};
    const std::string super{};
};

struct image_details
{
    image_details(const long width, const long height) : width(width), height(height) {}
    const long width{};
    const long height{};
};

int main(const int argc, const char** argv)
try
{
    if (argc != 2)
    {
        std::cout << "Please, pass the path to the original instances json file\n";
        return EXIT_FAILURE;
    }

    const std::string annotations_path(argv[1]);
    std::ifstream fin(annotations_path);
    if (not fin.good())
        throw std::runtime_error("ERROR while trying to open " + annotations_path + " file.");

    std::string set;
    if (annotations_path.find("train") != std::string::npos)
        set = "train";
    else if (annotations_path.find("val") != std::string::npos)
        set = "val";
    else
        throw std::runtime_error("Unsuported file name: it must contain either train or val.");

    std::cout << "Found COCO 2027 " << set << " set" << '\n';
    json annotations;
    fin >> annotations;

    image_dataset_metadata::dataset dataset;
    dataset.name = "COCO 2017 detection dataset";
    dataset.comment = set;

    // Parse the COCO categories
    std::map<int, category> categories;
    for (const auto& cat : annotations["categories"])
    {
        const auto id = cat["id"].get<int>();
        const auto name = cat["name"].get<std ::string>();
        const auto supercategory = cat["supercategory"].get<std::string>();
        categories.emplace(id, category(name, supercategory));
    }
    std::cout << "Number of categories: " << categories.size() << '\n';
    for (const auto& [id, c] : categories)
        std::cout << "id: " << id << ", name: " << c.name << ", super: " << c.super << '\n';

    // Parse the image sizes
    std::map<int, image_details> image_sizes;
    for (const auto& annot : annotations["images"])
    {
        const auto id = annot["id"].get<int>();
        const auto width = annot["width"].get<long>();
        const auto height = annot["height"].get<long>();
        image_sizes.emplace(id, image_details(width, height));
    }

    // Parse the bounding boxes
    std::map<int, std::vector<image_dataset_metadata::box>> image_boxes;
    for (const auto& annot : annotations["annotations"])
    {
        const auto bbox = annot["bbox"];
        DLIB_CASSERT(bbox.size() == 4);
        image_dataset_metadata::box box;
        box.rect.left() = std::round(bbox[0].get<double>());
        box.rect.top() = std::round(bbox[1].get<double>());
        box.rect.right() = std::round(bbox[0].get<double>() + bbox[2].get<double>());
        box.rect.bottom() = std::round(bbox[1].get<double>() + bbox[3].get<double>());
        const auto category_id = annot["category_id"].get<int>();
        const auto image_id = annot["image_id"].get<int>();
        box.label = categories.at(category_id).name;
        image_boxes[image_id].push_back(box);
    }

    // Convert the COCO dataset into XML
    for (const auto& [image_id, boxes] : image_boxes)
    {
        image_dataset_metadata::image image;
        std::ostringstream sout;
        sout << set << "2017/" << std::setw(12) << std::setfill('0') << image_id << ".jpg";
        image.filename = sout.str();
        image.boxes = boxes;
        image.width = image_sizes.at(image_id).width;
        image.height = image_sizes.at(image_id).height;
        dataset.images.push_back(std::move(image));
    }
    image_dataset_metadata::save_image_dataset_metadata(dataset, "coco_" + set + "2017.xml");
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
