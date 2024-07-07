#include "raylib.h"
#include "rlgl.h"

#include "rt.h"
#include "rt_cuda.h"
#include "material.h"
#include "primitive.h"
#include "scene.h"
#include "camera.h"

#include <cuda_gl_interop.h>

const int window_width = 1920;
const int window_height = 1080;

const int image_width = 1280;
const int image_height = 720;

void init_scene()
{
    MaterialHandle mat{};
    mat = add_lambertian(Color3{0.5f, 0.5f, 0.5f});
    add_sphere(Point3{0, -1000, 0}, 1000, mat);

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = random_float();
            Point3 center(a + 0.9f*random_float(), 0.2f, b + 0.9f*random_float());

            if ((center - Point3(4, 0.2f, 0)).length() > 0.9f)
            {
                if (choose_mat < 0.8f) {
                    // diffuse
                    Color3 albedo = {
                        random_float() * random_float(), 
                        random_float() * random_float(), 
                        random_float() * random_float()
                    };
                    mat = add_lambertian(albedo);
                    add_sphere(Point3{center.x, center.y, center.z}, 0.2f, mat);
                } else if (choose_mat < 0.95f) {
                    // metal
                    Color3 albedo = {
                        random_float(0.5f, 1), 
                        random_float(0.5f, 1), 
                        random_float(0.5f, 1)
                    };
                    float fuzz = random_float(0, 0.5f);
                    mat = add_metal(albedo, fuzz);
                    add_sphere(Point3{center.x, center.y, center.z}, 0.2f, mat);
                } else {
                    // glass
                    mat = add_dielectric(Color3{1.0f, 1.0f, 1.0f}, 1.5f);
                    add_sphere(Point3{center.x, center.y, center.z}, 0.2f, mat);
                }
            }
        }
    }

    mat = add_dielectric(Color3{1.0f, 1.0f, 1.0f}, 1.5f);
    add_sphere(Point3{0, 1, 0}, 1.0f, mat);

    mat = add_lambertian(Color3{0.4f, 0.2f, 0.1f});
    add_sphere(Point3{-4, 1, 0}, 1.0f, mat);

    mat = add_metal(Color3{0.7f, 0.6f, 0.5f}, 0.0);
    add_sphere(Point3{4, 1, 0}, 1.0f, mat);

    material_update_to_device();
    sphere_update_to_device();
}

int main()
{
    InitWindow(window_width, window_height, "rt_cuda");

    //init_host_curand(1234ULL);
    init_device_curand(image_width, image_height);

    init_scene();

    Image black_img = GenImageColor(image_width, image_height, BLACK);
    Texture2D target = LoadTextureFromImage(black_img);
    UnloadImage(black_img);

    cudaGraphicsResource_t target_cuda_res;
    CUDA_CALL_CHECK(cudaGraphicsGLRegisterImage(&target_cuda_res, target.id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    // camera
    RtCamera rt_camera;
    //rt_camera.aspect_ratio = 16.0f / 9.0f;
    rt_camera.image_width = 1280;
    rt_camera.image_height = 720;
    rt_camera.samples_per_pixel = 500;
    //rt_camera.samples_per_pixel = 10;
    rt_camera.max_depth = 50;
    //rt_camera.max_depth = 10;
    rt_camera.vfov = 20;
    rt_camera.lookfrom = Point3(13, 2, 3);
    rt_camera.lookat = Point3(0, 0, 0);
    rt_camera.vup = Vec3(0, 1, 0);
    rt_camera.defocus_angle = 0.6f;
    rt_camera.focus_distance = 10.0f;
    rt_camera.update();

    RtCameraGPU h_rt_camera_gpu = rt_camera.get_camera_gpu();
    RtCameraGPU *d_rt_camera_gpu;

    cudaMalloc(&d_rt_camera_gpu, sizeof(RtCameraGPU));
    cudaMemcpy(d_rt_camera_gpu, &h_rt_camera_gpu, sizeof(RtCameraGPU), cudaMemcpyHostToDevice);

    int sample_count = 0;
    while(!WindowShouldClose())
    {
        if (sample_count < rt_camera.samples_per_pixel)
        {
            sample_count++;
            launch_rt_kernel(target_cuda_res, d_rt_camera_gpu, image_width, image_height, sample_count);
        }

        BeginDrawing();

        ClearBackground(DARKGRAY);

        DrawTextureRec(
            target, Rectangle{0, 0, float(image_width), float(image_height)}, 
            Vector2{0, 0}, WHITE);

        if (sample_count < rt_camera.samples_per_pixel)
        {
            DrawText(
                TextFormat("Rendering...(%i/%i)", sample_count, rt_camera.samples_per_pixel),
                16, rt_camera.image_height + 16,
                32, WHITE
            );
        }
        else
        {
            DrawText(
                TextFormat("Done(%i/%i)", sample_count, rt_camera.samples_per_pixel),
                16, rt_camera.image_height + 16,
                32, WHITE
            );
        }

        EndDrawing();
    }

    cudaFree(d_rt_camera_gpu);

    UnloadTexture(target);

    CloseWindow();

    return 0;
}