#include "scene.h"

#include "interval.cuh"

Scene h_scene{};
__device__ __constant__ Scene d_scene;

__host__ void add_sphere(Point3 center, float radius, MaterialHandle material)
{
    assert(h_scene.sphere_count < MAX_SPHERE_NUMBER);
    h_scene.spheres[h_scene.sphere_count] = {center, radius, material};
    h_scene.sphere_count++;
}

__host__ void sphere_update_to_device()
{
    CUDA_CALL_CHECK(cudaMemcpyToSymbol(d_scene, &h_scene, sizeof(Scene)));
}

__device__ bool ray_hit_scene(const RtRay& ray, const Interval& ray_t, RtHitRecord& rec)
{
    bool hit = false;
    float closest_hit_root = ray_t.max;
    for (int i = 0; i < d_scene.sphere_count; i++)
    {
        const Sphere& sphere = d_scene.spheres[i];
        if (ray_hit_sphere(sphere, ray, Interval(ray_t.min, closest_hit_root), rec))
        {
            hit = true;
            closest_hit_root = rec.t;
        }
    }
    return hit;
}