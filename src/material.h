#ifndef MATERIAL_H
#define MATERIAL_H

#define MAX_LAMBERTIAN_MATERIAL_NUMBER 500 
#define MAX_METAL_MATERIAL_NUMBER 500 
#define MAX_DIELECTRIC_MATERIAL_NUMBER 500

#include "rt.h"
#include "rt_ray.h"

struct LambertianMaterial
{
    Color3 albedo;
};

struct MetalMaterial
{
    Color3 albedo;
    float fuzz;
};

struct DielectricMaterial
{
    Color3 albedo;
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refractive_index;
};

struct LambertianArray
{
    int count;
    LambertianMaterial materials[MAX_LAMBERTIAN_MATERIAL_NUMBER];
};
struct MetalArray
{
    int count;
    MetalMaterial materials[MAX_METAL_MATERIAL_NUMBER];
};
struct DielectricArray
{
    int count;
    DielectricMaterial materials[MAX_DIELECTRIC_MATERIAL_NUMBER];
};

__host__ MaterialHandle add_lambertian(Color3 albedo);

__host__ MaterialHandle add_metal(Color3 albedo, float fuzz);

__host__ MaterialHandle add_dielectric(Color3 albedo, float refractive_index);

__host__ void material_update_to_device();

__device__ bool scatter_lambertian(const RtRay& incident_ray, const RtHitRecord& rec, 
    Color3& attenuation, RtRay& scattered_ray);

__device__ bool scatter_metal(const RtRay& incident_ray, const RtHitRecord& rec, 
    Color3& attenuation, RtRay& scattered_ray);

__device__ float reflectance(float cosine, float refraction_index);

__device__ bool scatter_dielectric(const RtRay& incident_ray, const RtHitRecord& rec, 
    Color3& attenuation, RtRay& scattered_ray);

__device__ bool scatter(const RtRay& incident_ray, const RtHitRecord& hit_record, 
    Color3& attenuation, RtRay& scattered_ray);

#endif