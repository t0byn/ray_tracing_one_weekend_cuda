#ifndef SCENE_H
#define SCENE_H

#include "rt.h"
#include "rt_ray.h"
#include "primitive.h"

#define MAX_SPHERE_NUMBER 500

struct Scene
{
    int sphere_count;
    Sphere spheres[MAX_SPHERE_NUMBER];
};

class Interval;

__host__ void add_sphere(Point3 center, float radius, MaterialHandle material);

__host__ void sphere_update_to_device();

__device__ bool ray_hit_scene(const RtRay& ray, const Interval& ray_t, RtHitRecord& rec);

#endif