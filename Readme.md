This is my attempt to rewrite [ray_tracing_in_one_weekend](https://github.com/t0byn/ray_tracing_in_one_weekend) using CUDA and raylib.
This is my first time coding in CUDA, so the code is rather sloppy and not optimal.

## build
**Note: You need to have CUDA installed on your machine in order to build the code.**
```
git submodule init
git submodule update
cmake -S . -B build
cmake --build build
```

## run
```
.\build\Debug\demo.exe
```