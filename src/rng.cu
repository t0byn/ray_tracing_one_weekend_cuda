#include "rng.h"

#include <assert.h>

#include <curand.h>
#include <curand_kernel.h>

curandGenerator_t h_curand_gen = NULL;
__device__ curandState d_curand_state[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT] = {0};
extern __shared__ curandState s_curand_state[];

//__host__ void init_host_curand(unsigned long long seed)
//{
//    if (h_curand_gen == NULL)
//    {
//        CURAND_CALL_CHECK(curandCreateGenerator(&h_curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
//    }
//    CURAND_CALL_CHECK(curandSetPseudoRandomGeneratorSeed(h_curand_gen, seed));
//}

__global__ void init_device_curand()
{
    int block_size = blockDim.x * blockDim.y;
    int thread_id = block_size * gridDim.x * blockIdx.y 
        + block_size * blockIdx.x 
        + blockDim.x * threadIdx.y
        + threadIdx.x;
    curand_init(0, thread_id, 0, &d_curand_state[thread_id]);
}

__host__ void init_device_curand(const int image_width, const int image_height)
{
    assert(image_width % 8 == 0);
    assert(image_height % 8 == 0);
    dim3 block_dim(8, 8);
    dim3 grid_dim(image_width / 8, image_height / 8);
    init_device_curand<<<grid_dim, block_dim>>>();
}

__device__ void load_curand_state()
{
    int block_size = blockDim.x * blockDim.y;
    int thread_id = block_size * gridDim.x * blockIdx.y 
        + block_size * blockIdx.x 
        + blockDim.x * threadIdx.y
        + threadIdx.x;
    int state_id = blockDim.x * threadIdx.y + threadIdx.x;
    s_curand_state[state_id] = d_curand_state[thread_id];
}

__device__ void update_curand_state()
{
    int block_size = blockDim.x * blockDim.y;
    int thread_id = block_size * gridDim.x * blockIdx.y 
        + block_size * blockIdx.x 
        + blockDim.x * threadIdx.y
        + threadIdx.x;
    int state_id = blockDim.x * threadIdx.y + threadIdx.x;
    d_curand_state[thread_id] = s_curand_state[state_id];
}

__host__ __device__ float random_float()
{
#if defined(__CUDA_ARCH__)
    int block_size = blockDim.x * blockDim.y;
    int thread_id = block_size * gridDim.x * blockIdx.y 
        + block_size * blockIdx.x 
        + blockDim.x * threadIdx.y
        + threadIdx.x;
    curandState local_state = d_curand_state[thread_id];
    float x = curand_uniform(&local_state);
    d_curand_state[thread_id] = local_state;
    return x;
/*
    int state_id = blockDim.x * threadIdx.y + threadIdx.x;
    curandState local_state = s_curand_state[state_id];
    float x = curand_uniform(&local_state);
    s_curand_state[state_id] = local_state;
    return x;
*/
#else
    float x;
    //CURAND_CALL_CHECK(curandGenerateUniform(h_curand_gen, &x, 1));
    x = (rand() / (RAND_MAX + 1.0f));
    return x;
#endif
}

__host__ __device__ float random_float(float min, float max)
{
    float x = min + (max - min) * random_float();
    return x;
}