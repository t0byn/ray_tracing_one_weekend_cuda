#include "material.h"

LambertianArray h_lambertian_array{};
MetalArray h_metal_array{};
DielectricArray h_dielectric_array{};

__device__ __constant__ LambertianArray d_lambertian_array;
__device__ __constant__ MetalArray d_metal_array;
__device__ __constant__ DielectricArray d_dielectric_array;

__host__ MaterialHandle add_lambertian(Color3 albedo)
{
    assert(h_lambertian_array.count < MAX_LAMBERTIAN_MATERIAL_NUMBER);
    h_lambertian_array.materials[h_lambertian_array.count] = {albedo};
    return MaterialHandle{
        LAMBERTIAN,
        h_lambertian_array.count++
    };
}

__host__ MaterialHandle add_metal(Color3 albedo, float fuzz)
{
    assert(h_metal_array.count < MAX_METAL_MATERIAL_NUMBER);
    h_metal_array.materials[h_metal_array.count] = {albedo, fuzz};
    return MaterialHandle{
        METAL,
        h_metal_array.count++
    };
}

__host__ MaterialHandle add_dielectric(Color3 albedo, float refractive_index)
{
    assert(h_dielectric_array.count < MAX_DIELECTRIC_MATERIAL_NUMBER);
    h_dielectric_array.materials[h_dielectric_array.count] = {albedo, refractive_index};
    return MaterialHandle{
        DIELECTRIC,
        h_dielectric_array.count++
    };
}

__host__ void material_update_to_device()
{
    CUDA_CALL_CHECK(cudaMemcpyToSymbol(d_lambertian_array, &h_lambertian_array, sizeof(LambertianArray)));
    CUDA_CALL_CHECK(cudaMemcpyToSymbol(d_metal_array, &h_metal_array, sizeof(MetalArray)));
    CUDA_CALL_CHECK(cudaMemcpyToSymbol(d_dielectric_array, &h_dielectric_array, sizeof(DielectricArray)));
}

__device__ bool scatter_lambertian(const RtRay& incident_ray, const RtHitRecord& rec, 
    Color3& attenuation, RtRay& scattered_ray)
{
    LambertianMaterial material = d_lambertian_array.materials[rec.material.index];
    Vec3 unit_vector = random_unit_vec3();
    Vec3 scatter_direction = rec.normal + unit_vector;

    if (scatter_direction.near_zero()) 
        scatter_direction = rec.normal;

    scattered_ray.origin = rec.p;
    scattered_ray.dir = scatter_direction;
    attenuation = material.albedo;
    return true;
}

__device__ bool scatter_metal(const RtRay& incident_ray, const RtHitRecord& rec, 
    Color3& attenuation, RtRay& scattered_ray)
{
    MetalMaterial material = d_metal_array.materials[rec.material.index];
    Vec3 reflect_direction = reflect(incident_ray.dir, rec.normal);
    // fuzz
    reflect_direction = reflect_direction + (material.fuzz * random_unit_vec3());

    scattered_ray.origin = rec.p;
    scattered_ray.dir = reflect_direction;
    attenuation = material.albedo;
    return (dot(reflect_direction, rec.normal) > 0);
}

__device__ float reflectance(float cosine, float refraction_index)
{
    // Christophe Schlick's approximation for reflectance
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ bool scatter_dielectric(const RtRay& incident_ray, const RtHitRecord& rec, 
    Color3& attenuation, RtRay& scattered_ray)
{
    DielectricMaterial material = d_dielectric_array.materials[rec.material.index];
    float ri = rec.front_face ? (1.0f / material.refractive_index) : material.refractive_index;
    Vec3 unit_incident = unit_vector(incident_ray.dir);
    float cos_theta = fminf(dot(-unit_incident, rec.normal), 1);
    float sin_theta = sqrtf(1 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1;
    Vec3 scatter_direction{};
    if (cannot_refract || reflectance(cos_theta, ri) > random_float())
    {
        scatter_direction = reflect(unit_incident, rec.normal);
    }
    else
    {
        scatter_direction = refract(unit_incident, rec.normal, ri);
    }

    scattered_ray.origin = rec.p;
    scattered_ray.dir = scatter_direction;
    attenuation = material.albedo;
    return true;
}

__device__ bool scatter(const RtRay& incident_ray, const RtHitRecord& hit_record, 
    Color3& attenuation, RtRay& scattered_ray)
{
    switch(hit_record.material.type)
    {
        case LAMBERTIAN:
            return scatter_lambertian(incident_ray, hit_record, attenuation, scattered_ray);
        case METAL:
            return scatter_metal(incident_ray, hit_record, attenuation, scattered_ray);
        case DIELECTRIC:
            return scatter_dielectric(incident_ray, hit_record, attenuation, scattered_ray);
    }
    return false;
}