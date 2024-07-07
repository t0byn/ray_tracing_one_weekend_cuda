#ifndef RT_RAY_H
#define RT_RAY_H

#include "rt.h"

struct RtHitRecord
{
    Point3 p;
    Vec3 normal;
    float t;
    bool front_face;
    MaterialHandle material;
};

struct RtRay
{
    Point3 origin;
    Vec3 dir;
};

__device__ Point3 ray_at(const RtRay& ray, const float t);

#endif