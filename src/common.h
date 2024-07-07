#ifndef COMMON_H
#define COMMON_H

#include <math.h>

#define MAX_IMAGE_WIDTH 1920
#define MAX_IMAGE_HEIGHT 1080

#define CUDA_CALL_CHECK(x) { \
    cudaError_t error = (x); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error at %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    } \
}

// constants

const float pi = 3.1415926535897932385f;

// utility function

inline float degree_to_radian(const float degree)
{
    return degree * pi / 180;
}

#endif