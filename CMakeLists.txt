cmake_minimum_required(VERSION 3.31)
# sm_89 = Ada (RTX 4070 SUPER). Must be set before project(), which otherwise picks a default.
# Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(ozaki LANGUAGES CXX CUDA)

# Inline -I flags in compile_commands.json so clangd can see the include paths.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)  # provides CUDA::cudart, CUDA::cublas; CCCL headers ship in the toolkit include dir

add_library(ozaki_warnings INTERFACE)
target_compile_options(ozaki_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --expt-relaxed-constexpr -lineinfo>)

# Header-only: kernels.cuh + ozaki.cuh are compiled by whichever .cu file includes them.
add_library(ozaki INTERFACE)
target_include_directories(ozaki INTERFACE include)
target_link_libraries(ozaki INTERFACE CUDA::cudart)

# Pure C++ (no CUDA): the CPU version of Ozaki scheme II.
add_library(ozaki_cpu src/ozaki_cpu.cpp)
target_include_directories(ozaki_cpu PUBLIC include)
target_link_libraries(ozaki_cpu PRIVATE ozaki_warnings)

add_subdirectory(bench)
