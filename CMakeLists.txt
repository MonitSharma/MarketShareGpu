cmake_minimum_required(VERSION 3.16.3)

project(markshare_gpu)

find_package(OpenMP REQUIRED)

add_executable (markshare_main src/main.cpp)

target_compile_features(markshare_main PUBLIC cxx_std_17)
target_link_libraries(markshare_main PRIVATE OpenMP::OpenMP_CXX -ltbb)