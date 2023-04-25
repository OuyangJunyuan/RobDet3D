cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDNN_ROOT /home/nrsl/software/cuda/cudnn/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/cuda)
set(CUDNN_LIBRARY_PATH  ${CUDNN_ROOT}/lib)
set(CUDNN_INCLUDE_PATH  ${CUDNN_ROOT}/include)
project(SimpleConcurrentGPUHashTable LANGUAGES CXX CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -rdc=true)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_35,code=sm_35")
file(GLOB SOURCE_FILES *.cu *.cpp)
set(CONDA_ENV /home/nrsl/software/anaconda3/envs/torch112)
set(Torch_DIR ${CONDA_ENV}/lib/python3.7/site-packages/torch/share/cmake/Torch/)
find_package(Python3 REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CONDA_ENV}/lib/python3.7/site-packages/torch/include/
        ${CONDA_ENV}/lib/python3.7/site-packages/torch/include/torch/csrc/api/include
        ${TORCH_INCLUDE_DIRS})
include_directories(
    ${CUDA_INCLUDE_DIRS}
    /home/nrsl/software/anaconda3/envs/torch112/include/python3.7m/)
add_library(hvcs_cuda SHARED ${SOURCE_FILES})

target_link_libraries( hvcs_cuda PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)