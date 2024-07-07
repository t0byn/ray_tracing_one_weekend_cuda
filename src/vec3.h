#ifndef VEC3_H
#define VEC3_H

#include "rng.h"

#include <cuda_runtime_api.h>

class Vec3
{
public:
    float x, y, z;

    //__host__ __device__ Vec3() : x{0}, y{0}, z{0} {};
    // We don't want the default constructor change member variable, 
    // because there are some vec3 resides in device constant memory space.
    // see https://stackoverflow.com/a/27042471
    __host__ __device__ Vec3() {};
    __host__ __device__ Vec3(float x, float y, float z) : x{x}, y{y}, z{z} {};

    __host__ __device__ Vec3 operator-() const;

    __host__ __device__ Vec3& operator+=(const Vec3& v);

    __host__ __device__ Vec3& operator*=(const float t);

    __host__ __device__ Vec3& operator/=(const float t);

    __host__ __device__ float length_squared() const;

    __host__ __device__ float length() const;

    __host__ __device__ bool near_zero() const;

    __host__ __device__ static Vec3 random();

    __host__ __device__ static Vec3 random(const float min, const float max);
};

using Point3 = Vec3;

using Color3 = Vec3;

__host__ __device__ Vec3 operator+(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 operator-(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 operator*(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 operator*(const float t, const Vec3& v);

__host__ __device__ Vec3 operator*(const Vec3& v, const float t);

__host__ __device__ Vec3 operator/(const Vec3& v, const float t);

__host__ __device__ float dot(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 unit_vector(const Vec3& v);

__host__ __device__ Vec3 random_unit_vec3();

__host__ __device__ Vec3 reflect(const Vec3& incident, const Vec3& normal);

__host__ __device__ Vec3 refract(const Vec3& unit_incident, const Vec3& normal, const float eta_over_eta_prime);

#endif