#ifndef INTERVAL_CUH
#define INTERVAL_CUH

#include "common.h"

const float infinity = INFINITY;

class Interval
{
public:
    float min;
    float max;

    __device__ Interval() : min(+infinity), max(-infinity) {};
    __device__ Interval(float min, float max) : min(min), max(max) {};

    __device__ bool contains(const float x) const
    {
        return (x >= min) && (x <= max);
    }

    __device__ bool surrounds(const float x) const
    {
        return (x > min) && (x < max);
    }

    __device__ float clamp(const float x) const
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
};

#endif