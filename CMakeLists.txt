cmake_minimum_required(VERSION 3.16.3)

project(markshare_gpu)

# Find OpenMP
find_package(OpenMP REQUIRED)
# Find CUDA Toolkit
find_package(CUDAToolkit)

add_subdirectory(external/argparse)

add_executable (markshare_main src/main.cpp)

# If CUDA is found, add definitions and link CUDA libraries
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA was found")
    target_compile_definitions(markshare_main PRIVATE WITH_GPU)
    target_link_libraries(markshare_main PRIVATE CUDA::cudart)
else(CUDAToolkit_FOUND)
    message(STATUS "CUDA not found")
endif(CUDAToolkit_FOUND)

target_compile_features(markshare_main PUBLIC cxx_std_17)

target_include_directories(markshare_main PRIVATE external/argparse/include/argparse)
target_link_libraries(markshare_main PRIVATE OpenMP::OpenMP_CXX -ltbb)