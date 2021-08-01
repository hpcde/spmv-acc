# spmv-acc

HIP acceleration for SpMV solver.

## Build
### Pre-requirements
- [ROCM](https://rocmdocs.amd.com): version 3.x or higher. For example: `module load compiler/rocm/3.9.1`
- [HIP](https://github.com/ROCm-Developer-Tools/HIP)
- [CMake](https://cmake.org): version 3.6 or higher.

### Build steps
- Build and verify on GPU side (make sure lib `rocsparse` is loaded):
```bash
CC=clang CXX=hipcc cmake -DDEVICE_SIDE_VERIFY_FLAG=ON -DCMAKE_BUILD_TYPE=Release -B./build-hip -S./
cmake --build ./build-hip
./build-hip/bin/spmv-hip examples/data/rajat03.csr
```

- Build and verify on CPU side:
```bash
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -B./build-hip -S./
cmake --build ./build-hip
./build-hip/bin/spmv-hip examples/data/rajat03.csr
```

- Build by specifying a kernel strategy (e.g., use strategy `WF_ROW`):
```bash
CC=clang CXX=hipcc cmake -DKERNEL_STRATEGY=WF_ROW -DCMAKE_BUILD_TYPE=Release -B./build-hip-wf-row -S./
cmake --build ./build-hip-wf-row
./build-hip-wf-row/bin/spmv-hip examples/data/rajat03.csr
```

## For Developers
### Add a new kernel strategy
A **kernel strategy** is an algorithm for calculating SpMV on device side.  
You can specific another kernel strategy (algorithm) by following rules:
1. Edit [src/configure.cmake](src/configure.cmake) to add a kernel strategy checking (e.g. add a strategy named `awesome_spmv`).
   ```diff
   +elseif (KERNEL_STRATEGY_LOWER MATCHES "awesome_spmv")
   +    set(KERNEL_STRATEGY_AWESOME_SPMV ON)
   else ()
       MESSAGE(FATAL_ERROR "unsupported kernel strategy ${KERNEL_STRATEGY}")
   endif ()
   ```
2. Edit [src/building_config.h.in](src/building_config.h.in) for generating C/C++ **Macro** defines of the corresponding strategy.
   ```diff
   #cmakedefine KERNEL_STRATEGY_DEFAULT
   +#cmakedefine KERNEL_STRATEGY_AWESOME_SPMV
   ```
3. Edit file `src/acc/CMakeLists.txt` and add the kernel strategy name, then CMake can find the source files of the kernel strategy.
   e.g.,
   ```diff
   # all enabled strategies
   set(ENABLED_STRATEGIES
       default
   +   awesome_spmv
   ```
4. Create a new directory named `hip-awesome-spmv` (replace '_' in strategy name to '-') under `src/acc` directory 
   and place your code for the new strategy to this directory.

5. Add file `source_list.cmake` to directory `src/acc/hip-awesome-spmv` to include the source files of the new strategy.
    Please refer to file `src/acc/hip/source_list.cmake` for more details.

6. Edit file `src/acc/strategy_picker.cpp` to call the entry function of the corresponding strategy.
   e.g.,
   ```diff
   void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                    const double *value, const double *x, double *y) {
   #ifdef KERNEL_STRATEGY_DEFAULT
   default_sparse_spmv(trans, alpha, beta, m, n, rowptr, colindex, value, x, y);
   #endif
   +#ifdef KERNEL_STRATEGY_AWESOME_SPMV
   +awesome_sparse_spmv(trans, alpha, beta, m, n, rowptr, colindex, value, x, y);
   +#endif
   ```
