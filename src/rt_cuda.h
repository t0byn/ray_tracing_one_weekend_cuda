#ifndef RT_CUDA_H
#define RT_CUDA_H

#include "glad.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "camera.h"

void launch_rt_kernel(cudaGraphicsResource_t target, RtCameraGPU* rt_camera_gpu, int image_width, int image_height, int sample);

#endif