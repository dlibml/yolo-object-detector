cmake_minimum_required(VERSION 3.14)
project("yolo" LANGUAGES CXX)

set(CPACK_PACKAGE_NAME "yolo")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

# Use C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Colored warnings
option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)
if(${FORCE_COLORED_OUTPUT})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

# Enable ccache if it exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

find_program(MOLD_FOUND mold)
if(MOLD_FOUND)
    add_compile_options(-fuse-ld=mold)
    set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Optimization flags
include(CheckCXXCompilerFlag)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

find_package(OpenCV REQUIRED)

# Dependency management
include(FetchContent)
# set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)
set(FETCHCONTENT_QUIET FALSE)
macro(fetch_content name tag repository)
    FetchContent_Declare(
        ${name}
        GIT_REPOSITORY ${repository}
        GIT_TAG        ${tag}
        GIT_PROGRESS   TRUE
        GIT_SHALLOW    TRUE
        SOURCE_DIR     ${PROJECT_SOURCE_DIR}/external/${name}
    )
    message(STATUS "Fetching ${name} ${tag}")
    FetchContent_MakeAvailable(${name})
endmacro()

macro(add_dlib_executable name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE dlib::dlib)
    target_include_directories(${name} PRIVATE src)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    install(TARGETS ${name} DESTINATION bin)
endmacro()

macro(add_dlib_library name)
    add_library(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE dlib::dlib)
    target_include_directories(${name} PRIVATE src)
    target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic -Wno-deprecated-copy)
    install(TARGETS ${name} DESTINATION lib)
endmacro()

fetch_content(dlib 79e2d1373add8d9e265d6a16c4952f5273600e97 https://github.com/davisking/dlib.git)
fetch_content(json v3.12.0 https://github.com/nlohmann/json.git)

add_compile_options(-ftemplate-depth=2000)

add_dlib_executable(compute_anchors)

add_dlib_library(yolo_logo)
add_dlib_library(draw)
add_dlib_library(webcam_window)

add_dlib_library(model)
add_dlib_library(sgd_trainer)
add_dlib_library(detector_utils)
add_dlib_library(metrics PRIVATE model detector_utils)

add_dlib_executable(train)
target_link_libraries(train PRIVATE model sgd_trainer metrics detector_utils)

add_dlib_executable(test)
target_link_libraries(test PRIVATE model sgd_trainer metrics detector_utils)

add_dlib_executable(detect)
target_link_libraries(detect PRIVATE model sgd_trainer detector_utils draw webcam_window yolo_logo ${OpenCV_LIBS})
target_include_directories(detect PRIVATE ${OpenCV_INCLUDE_DIRS})

add_dlib_executable(fuse)
target_link_libraries(fuse PRIVATE model sgd_trainer)

add_dlib_executable(coco2xml)
target_link_libraries(coco2xml PRIVATE nlohmann_json::nlohmann_json)
add_dlib_executable(xml2coco)
target_link_libraries(xml2coco PRIVATE nlohmann_json::nlohmann_json)

add_dlib_executable(convert_images)
add_dlib_executable(xml2darknet)
add_dlib_executable(darknet2xml)
add_dlib_executable(draw_boxes)
target_link_libraries(draw_boxes PRIVATE draw)

add_dlib_executable(evalcoco)
target_link_libraries(evalcoco PRIVATE model detector_utils draw nlohmann_json::nlohmann_json)

