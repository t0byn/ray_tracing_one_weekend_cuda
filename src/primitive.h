#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "rt.h"
#include "rt_ray.h"

struct Sphere
{
    Point3 center;
    float radius;
    MaterialHandle material;
};

class Interval;

__device__ bool ray_hit_sphere(const Sphere& sphere, const RtRay& ray, const Interval& ray_t, RtHitRecord& rec);

#endif