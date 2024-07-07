#include "vec3.h"

__host__ __device__ Vec3 Vec3::operator-() const
{
    return Vec3(-x, -y, -z);
}

__host__ __device__ Vec3& Vec3::operator+=(const Vec3& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const float t)
{
    x *= t;
    y *= t;
    z *= t;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator/=(const float t)
{
    *this *= (1.f/t);
    return *this;
}

__host__ __device__ float Vec3::length_squared() const
{
    return (x*x + y*y + z*z);
}

__host__ __device__ float Vec3::length() const
{
    return sqrtf(length_squared());
}

__host__ __device__ bool Vec3::near_zero() const
{
    float s = float(1e-8);
    return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
}

__host__ __device__ Vec3 Vec3::random()
{
    return Vec3(random_float(), random_float(), random_float());
}

__host__ __device__ Vec3 Vec3::random(const float min, const float max)
{
    return Vec3(random_float(min, max), random_float(min, max), random_float(min, max));
}

__host__ __device__ Vec3 operator+(const Vec3& u, const Vec3& v)
{
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ Vec3 operator-(const Vec3& u, const Vec3& v)
{
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ Vec3 operator*(const Vec3& u, const Vec3& v)
{
    return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ Vec3 operator*(const float t, const Vec3& v)
{
    return Vec3(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ Vec3 operator*(const Vec3& v, const float t)
{
    return t * v;
}

__host__ __device__ Vec3 operator/(const Vec3& v, const float t)
{
    return (1.f/t) * v;
}

__host__ __device__ float dot(const Vec3& u, const Vec3& v)
{
    return (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
}

__host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v)
{
    return Vec3(
        (u.y * v.z) - (u.z * v.y),
        (u.z * v.x) - (u.x * v.z),
        (u.x * v.y) - (u.y * v.x)
    );
}

__host__ __device__ Vec3 unit_vector(const Vec3& v)
{
    return (v / v.length());
}

__host__ __device__ Vec3 random_unit_vec3()
{
    Vec3 v = Vec3::random(-1.f, 1.f);
    return unit_vector(Vec3(tanf(v.x), tanf(v.y), tanf(v.z)));
}

__host__ __device__ Vec3 reflect(const Vec3& incident, const Vec3& normal)
{
    Vec3 reflected = (incident - 2 * dot(incident, normal) * normal);
    return reflected;
}

__host__ __device__ Vec3 refract(const Vec3& unit_incident, const Vec3& normal, const float eta_over_eta_prime)
{
    float cos_theta = fminf(dot(-unit_incident, normal), 1.0f);
    Vec3 perpendicular = 
        eta_over_eta_prime * (unit_incident + cos_theta * normal);
    Vec3 parallel = 
        -sqrtf(1 - perpendicular.length_squared()) * normal;
    Vec3 refracted = perpendicular + parallel;
    return refracted;
}