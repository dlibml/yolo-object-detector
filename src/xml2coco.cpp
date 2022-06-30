#include <dlib/data_io.h>
#include <nlohmann/json.hpp>

using namespace dlib;
using json = nlohmann::json;

int main(const int argc, const char** argv)
try
{
    if (argc != 2)
    {
        std::cout << "Please, pass the path to an XML dataset file\n";
        return EXIT_FAILURE;
    }

    image_dataset_metadata::dataset dataset;
    image_dataset_metadata::load_image_dataset_metadata(dataset, argv[1]);

    std::set<std::string> labels_set;
    for (const auto& image : dataset.images)
        for (const auto& box : image.boxes)
            labels_set.insert(box.label);

    std::map<std::string, int> category_map;
    {
        int i = 0;
        for (const auto& label : labels_set)
            category_map.emplace(label, i++);
    }

    json data;

    std::cout << "Number of categories: " << category_map.size() << '\n';
    for (const auto& [label, id] : category_map)
    {
        std::cout << std::setw(3) << id << ": " << label << '\n';
        data["categories"].push_back(json{{"id", id}, {"name", label}, {"supercategory", ""}});
    }

    data["info"] = json{
        {"year", 1970},
        {"version", "v1.0.0"},
        {"description", dataset.name},
        {"contributor", ""},
        {"url", ""},
        {"date_created", ""}};

    for (size_t i = 0; i < dataset.images.size(); ++i)
    {
        const auto& image = dataset.images.at(i);
        data["images"].emplace_back(json{
            {"id", i},
            {"width", image.width},
            {"height", image.height},
            {"file_name", image.filename}});

        static int box_id = 0;
        for (const auto& box : image.boxes)
        {
            const auto& r = box.rect;
            data["annotations"].push_back(json{
                {"id", box_id++},
                {"image_id", i},
                {"category_id", category_map.at(box.label)},
                {"iscrowd", 0},
                {"area", r.area()},
                {"bbox", json{r.left(), r.top(), r.width(), r.height()}}});
        }
    }

    std::ofstream fout("dataset.json");
    fout << data.dump(2);
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}
