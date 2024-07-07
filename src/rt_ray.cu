#include "rt_ray.h"

__device__ Point3 ray_at(const RtRay& ray, const float t)
{
    return ray.origin + (t * ray.dir);
}