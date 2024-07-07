#ifndef MATERIAL_TYPE_H
#define MATERIAL_TYPE_H

enum MaterialType
{
    UNDEFINED,
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
};

struct MaterialHandle
{
    MaterialType type;
    int index;
};

#endif