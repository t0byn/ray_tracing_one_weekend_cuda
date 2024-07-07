cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# enable this for Separate Compilation 
# see https://stackoverflow.com/questions/31006581/cuda-device-unresolved-extern-function
# also see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-options-for-separate-compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(ray_tracing_one_weekend_cuda LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_GENERATOR MATCHES "Visual Studio")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:MSVC>:/utf-8>>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/utf-8>>")
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

#add_compile_definitions(RLGL_SHOW_GL_DETAILS_INFO)

set(OPENGL_VERSION "4.3" CACHE STRING "")

list(APPEND HEADERS src/glad.h
                    src/common.h
                    src/material_type.h
                    src/rng.h
                    src/vec3.h
                    src/rt.h
                    src/rt_ray.h
                    src/primitive.h
                    src/material.h
                    src/camera.h
                    src/scene.h
                    src/interval.cuh
                    src/rt_cuda.h)

list(APPEND SOURCES src/main.cc
                    src/rt_ray.cu
                    src/rng.cu
                    src/vec3.cu
                    src/primitive.cu
                    src/material.cu
                    src/scene.cu
                    src/rt_cuda.cu)

add_executable(demo ${SOURCES} ${HEADERS})

set_target_properties(
  demo PROPERTIES 
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_subdirectory("raylib")

target_link_libraries(demo PUBLIC raylib)
find_package(CUDAToolkit)
target_link_libraries(demo PUBLIC CUDA::curand)

if(MSVC)
  target_link_options(demo PUBLIC /NODEFAULTLIB:libcmt.lib)
endif()