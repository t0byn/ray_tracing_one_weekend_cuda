#include "rt_cuda.h"

#include "rt.h"
#include "rt_ray.h"
#include "interval.cuh"
#include "scene.h"
#include "material.h"

#include <curand_kernel.h>

__device__ float3 color_accumulation[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];

__device__ Vec3 raycast(RtCameraGPU* camera, const RtRay& ray)
{
    RtRay send_ray = ray;
    Vec3 color = Vec3(1.f, 1.f, 1.f);
    for (int depth = 0; depth <= camera->max_depth; depth++)
    {
        if (depth == camera->max_depth)
        {
            color = color * Vec3(0.f, 0.f, 0.f);
            break;
        }

        RtHitRecord rec;
        if (ray_hit_scene(send_ray, Interval(0.001f, +INFINITY), rec))
        {
            Vec3 attenuation;
            RtRay scatterred_ray;
            if (scatter(send_ray, rec, attenuation, scatterred_ray))
            {
                color = color * attenuation;
                send_ray = scatterred_ray;
            }
            else
            {
                color = color * Vec3(0.f, 0.f, 0.f);
                break;
            }
        }
        else
        {
            Vec3 white = Vec3(1.f, 1.f, 1.f);
            Vec3 blue = Vec3(0.5f, 0.7f, 1.0f);
            Vec3 unit_dir = unit_vector(send_ray.dir);
            float a = 0.5f * (unit_dir.y + 1.0f);
            Vec3 sky_color = (1 - a) * white + a * blue;
            color = color * sky_color;
            break;
        }
    }
    return color;
}

// gamma correction with gamma = 2
__device__ void linear_to_gamma(Vec3 &color)
{
    if (color.x > 0) color.x = sqrtf(color.x);
    if (color.y > 0) color.y = sqrtf(color.y);
    if (color.z > 0) color.z = sqrtf(color.z);
    if (color.x < 0.0f) color.x = 0.0f;
    if (color.x > 0.999f) color.x = 0.999f;
    if (color.y < 0.0f) color.y = 0.0f;
    if (color.y > 0.999f) color.y = 0.999f;
    if (color.z < 0.0f) color.z = 0.0f;
    if (color.z > 0.999f) color.z = 0.999f;
}

__global__ void rt_kernel(cudaSurfaceObject_t target_surface, RtCameraGPU* camera, int sample)
{
    //load_curand_state();

    int pixel_x = blockDim.x * blockIdx.x + threadIdx.x;
    int pixel_y = blockDim.y * blockIdx.y + threadIdx.y;
    Vec3 sample_offset = Vec3(random_float() - 0.5f, random_float() - 0.5f, 0.f);
    Vec3 pixel_sample_point = camera->pixel_00
        + (pixel_x + sample_offset.x) * camera->pixel_delta_u
        + (pixel_y + sample_offset.y) * camera->pixel_delta_v;
    Vec3 origin_offset = Vec3(random_float() - 0.5f, random_float() - 0.5f, 0.f);
    Vec3 ray_origin = camera->lookfrom
        + (origin_offset.x * camera->defocus_disk_u)
        + (origin_offset.y * camera->defocus_disk_v);
    Vec3 ray_direction = pixel_sample_point - ray_origin;

    RtRay ray;
    ray.origin = ray_origin;
    ray.dir = ray_direction;

    Vec3 color = raycast(camera, ray);

    int block_size = blockDim.x * blockDim.y;
    int thread_id = block_size * gridDim.x * blockIdx.y 
        + block_size * blockIdx.x 
        + blockDim.x * threadIdx.y
        + threadIdx.x;

    color_accumulation[thread_id] = float3 {
        color.x + color_accumulation[thread_id].x, 
        color.y + color_accumulation[thread_id].y, 
        color.z + color_accumulation[thread_id].z
    };

    Vec3 pixel_color = Vec3 {
        color_accumulation[thread_id].x / sample,
        color_accumulation[thread_id].y / sample,
        color_accumulation[thread_id].z / sample
    };

    linear_to_gamma(pixel_color);

    uchar4 color_bytes;
    color_bytes.x = unsigned char(pixel_color.x * 255U);
    color_bytes.y = unsigned char(pixel_color.y * 255U);
    color_bytes.z = unsigned char(pixel_color.z * 255U);
    color_bytes.w = (unsigned char)255U;
    surf2Dwrite(color_bytes, target_surface, pixel_x * sizeof(uchar4), pixel_y, cudaBoundaryModeClamp);

    //update_curand_state();
}

void launch_rt_kernel(cudaGraphicsResource_t target, RtCameraGPU* rt_camera_gpu, int image_width, int image_height, int sample)
{
    if (sample == 1)
    {
        float3* d_color_accumulation;
        CUDA_CALL_CHECK(cudaGetSymbolAddress((void **)&d_color_accumulation, color_accumulation));
        CUDA_CALL_CHECK(cudaMemset(d_color_accumulation, 0, sizeof(float3) * MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT));
    }

/*
    static cudaArray_t target_mapped_array;
    static cudaResourceDesc target_surface_desc;
    static cudaSurfaceObject_t target_surface;
    if (sample == 1)
    {
        cudaGraphicsMapResources(1, &target, NULL);
        CUDA_CALL_CHECK(cudaGraphicsSubResourceGetMappedArray(&target_mapped_array, target, 0, 0));

        memset(&target_surface_desc, 0, sizeof(cudaResourceDesc));
        target_surface_desc.resType = cudaResourceTypeArray;
        target_surface_desc.res.array.array = target_mapped_array;

        CUDA_CALL_CHECK(cudaCreateSurfaceObject(&target_surface, &target_surface_desc));
    }
*/

    cudaGraphicsMapResources(1, &target, NULL);

    cudaArray_t target_mapped_array;
    CUDA_CALL_CHECK(cudaGraphicsSubResourceGetMappedArray(&target_mapped_array, target, 0, 0));

    cudaResourceDesc target_surface_desc;
    memset(&target_surface_desc, 0, sizeof(cudaResourceDesc));
    target_surface_desc.resType = cudaResourceTypeArray;
    target_surface_desc.res.array.array = target_mapped_array;

    cudaSurfaceObject_t target_surface;
    CUDA_CALL_CHECK(cudaCreateSurfaceObject(&target_surface, &target_surface_desc));

    dim3 block_dim(8, 8);
    dim3 grid_dim(image_width / 8, image_height / 8);
    rt_kernel<<<grid_dim, block_dim, block_dim.x * block_dim.y>>>(target_surface, rt_camera_gpu, sample);
    //rt_kernel<<<grid_dim, block_dim, block_dim.x * block_dim.y * sizeof(curandState)>>>(target_surface, rt_camera_gpu, sample);

    cudaDeviceSynchronize();

    CUDA_CALL_CHECK(cudaDestroySurfaceObject(target_surface));
    CUDA_CALL_CHECK(cudaGraphicsUnmapResources(1, &target, NULL));
}