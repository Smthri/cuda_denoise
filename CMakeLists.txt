cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cuda_denoise LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_denoise main.cu weights.h)

set_target_properties(cuda_denoise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

include(FindOpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuda_denoise ${OpenCV_LIBS})

if (BLOCK_SIZE)
    add_definitions(-DBLOCK_SIZE=${BLOCK_SIZE})
else()
    add_definitions(-DBLOCK_SIZE=10)
endif()
