# spmv-acc

HIP acceleration for SpMV solver.

## Build
### Pre-requestment
- [ROCM](https://rocmdocs.amd.com): version 3.x or higher. For example: `module load compiler/rocm/3.9.1`
- [HIP](https://github.com/ROCm-Developer-Tools/HIP)
- [CMake](https://cmake.org): version 3.6 or higher.

### Build steps
Build and verify on GPU side (make sure lib `rocsparse` is loaded):
```bash
CC=clang CXX=hipcc cmake -DDEVICE_SIDE_VERIFY_FLAG=ON -DCMAKE_BUILD_TYPE=Release -B./build-hip -S./
cmake --build ./build-hip
./build-hip/bin/spmv-hip
```

Build and verify on CPU side:
```bash
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -B./build-hip -S./
cmake --build ./build-hip
./build-hip/bin/spmv-hip
```
