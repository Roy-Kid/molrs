cmake_minimum_required(VERSION 3.18)
project(molrs_core_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_library(molrs_core_cuda STATIC
    cpp/device_runtime.cpp
    cpp/potentials/potential.cpp
    cpp/potentials/pair_lj126.cpp
    cpp/potentials/bond_harmonic.cpp
    cuda/bond_harmonic.cu
    cuda/angle_harmonic.cu
    cuda/pair_lj126.cu
    cuda/reduce.cu
    cuda/neighborlist.cu
    cuda/simbox_gpu.cu
)

target_include_directories(molrs_core_cuda PUBLIC include cuda/include cpp/include cuda)
target_link_libraries(molrs_core_cuda CUDA::cudart)

install(TARGETS molrs_core_cuda ARCHIVE DESTINATION lib)
