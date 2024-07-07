#include "primitive.h"

#include "interval.cuh"

__device__ bool ray_hit_sphere(const Sphere& sphere, const RtRay& ray, const Interval& ray_t, RtHitRecord& rec)
{
    Vec3 oc = sphere.center - ray.origin;
    float a = ray.dir.length_squared();
    float h = dot(ray.dir, oc);
    float c = oc.length_squared() - (sphere.radius * sphere.radius);
    float discriminant = (h * h) - (a * c);

    if (discriminant < 0)
    {
        return false;
    }

    float sqrt_d = sqrtf(discriminant);

    // find the nearest root that lies in the acceptable range
    float root = (h - sqrt_d) / a;
    if (!ray_t.surrounds(root))
    {
        root = (h + sqrt_d) / a;
        if (!ray_t.surrounds(root)) return false;
    }

    rec.t = root;
    rec.p = ray_at(ray, root);
    Vec3 outward_normal = (rec.p - sphere.center) / sphere.radius;
    rec.front_face = dot(ray.dir, outward_normal) <= 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
    rec.material = sphere.material;

    return true;
}