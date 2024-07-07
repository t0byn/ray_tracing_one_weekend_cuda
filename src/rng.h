#ifndef RNG_H
#define RNG_H

#include "common.h"

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#define CURAND_CALL_CHECK(x) { if ((x) != CURAND_STATUS_SUCCESS) { fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); } }

//__host__ void init_host_curand(unsigned long long seed);
__host__ void init_device_curand(const int image_width, const int image_height);

__device__ void load_curand_state();
__device__ void update_curand_state();

__host__ __device__ float random_float();
__host__ __device__ float random_float(float min, float max);

#endif