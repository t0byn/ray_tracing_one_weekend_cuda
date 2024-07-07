#ifndef CAMERA_H
#define CAMERA_H

#include "rt.h"

// the camera struct used in GPU
struct RtCameraGPU
{
    Vec3 lookfrom;
    Vec3 pixel_00;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 defocus_disk_u;
    Vec3 defocus_disk_v;
    int max_depth;
    int samples_per_pixel;
};

class RtCamera
{
public:
    int image_width = 100;
    int image_height = 100;
    int samples_per_pixel = 10;
    int max_depth = 10;

    float vfov = 90.0f; // vertical view angle (field of view)
    Point3 lookfrom = Point3(0, 0, 0);
    Point3 lookat = Point3(0, 0, -1);
    Vec3 vup = Vec3(0, 1, 0);

    float defocus_angle = 0.0f;
    float focus_distance = 10.0f;

    void update()
    {
        // image
        //image_height = int(image_width / aspect_ratio);
        //if (image_height < 1) image_height = 1; // make sure image height at least 1
        aspect_ratio = float(image_width) / float(image_height);

        // camera
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // viewport
        float theta = degree_to_radian(vfov);
        float h = tanf(theta / 2.0f) * focus_distance;
        float viewport_height = 2.0f * h;
        float viewport_width = viewport_height * aspect_ratio;
        Vec3 viewport_u = u * viewport_width;
        Vec3 viewport_v = -v * viewport_height;

        // pixel spacing
        pixel_delta_u = viewport_u / float(image_width);
        pixel_delta_v = viewport_v / float(image_height);

        Point3 viewport_top_left = 
            lookfrom - (focus_distance * w) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel_00 = viewport_top_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        pixel_samples_scale = 1.0f / samples_per_pixel;

        float disk_radius = tanf(degree_to_radian(defocus_angle / 2.0f)) * focus_distance;
        defocus_disk_u = u * disk_radius;
        defocus_disk_v = v * disk_radius;
    }

    RtCameraGPU get_camera_gpu()
    {
        RtCameraGPU cam_gpu;
        cam_gpu.lookfrom = lookfrom;
        cam_gpu.pixel_00 = pixel_00;
        cam_gpu.pixel_delta_u = pixel_delta_u;
        cam_gpu.pixel_delta_v = pixel_delta_v;
        cam_gpu.defocus_disk_u = defocus_disk_u;
        cam_gpu.defocus_disk_v = defocus_disk_v;
        cam_gpu.max_depth = max_depth;
        cam_gpu.samples_per_pixel = samples_per_pixel;
        return cam_gpu;
    }

private:
    float aspect_ratio;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Point3 pixel_00;
    float pixel_samples_scale;
    Vec3 u, v, w;
    Vec3 defocus_disk_u, defocus_disk_v;
};

#endif