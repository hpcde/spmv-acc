<a name="unreleased"></a>
## [Unreleased]


<a name="v0.3.0"></a>
## [v0.3.0] - 2021-07-15
### Build
- **cmake:** add ability to config CU number for kernel-line strategy and generate global config

### Chore
- update source files created authors and created date

### Feat
- use AVAILABLE_CU in cmake generated `building_config.h` to config LDS size of Block
- **kernel-flat:** add support for multiple rounds of csr_val*vec_x in a Block loop
- **kernel-flat:** make algorithm compatible with the case of "one row is cut by more than 2 blocks"
- **kernel-flat:** add base implementation of `flat` strategy
- **kernel-thread-row:** fallback to use native thread-row method when the nnz per row is large
- **kernel-thread-row:** new thread-row strategy, but with more coherent memory access
- **line:** add implementation of kernel strategy `line`

### Fix
- **kernel-line:** fix incorrect computation of variable `block_end_row_id`
- **kernel-line:** fix incorrect reduction when threads number in Block is less than rows processed

### Merge
- **cmake:** Merge branch 'feature-cmake-config-CUs' into 'main'
- **kernel-flat:** Merge branch 'feature-kernel-flat' into 'main'
- **kernel-line:** Merge branch 'feature-kernel-line' into 'main'
- **kernel-thread-row:** Merge branch 'opt-thread-row-for-small-row-2' into 'main'
- **kernel-thread-row:** Merge branch 'opt-thread-row-for-small-row' into 'main'

### Perf
- **kernel-line:** use `min`(instead of `if`) to obtain array index for matrix-vector multiplication
- **kernel-thread-row:** replace memory access of wavefront row start/end with broadcast(`__shfl`)
- **kernel-thread-row:** use nontemporal load and store to load/store y vector from/to device mmory
- **kernel-thread-row:** load 2 matrix values and column indexes in each loop for multiplication
- **kernel-thread-row:** tune block and grid dim to achieve better performance for new kernel func
- **kernel-thread-row:** remove unnecessary `if` in multiplication and reduction step

### Refactor
- **kernel-thread-row:** rename kernel func from device_sparse_spmv_acc to native_thread_row


<a name="v0.2.4"></a>
## [v0.2.4] - 2021-07-06
### Feat
- **utils:** add utils macro to count memory bandwidth

### Merge
- **kernel-vector-row:** Merge branch 'opt-vector-pipeline' into 'main'
- **utils:** Merge branch 'benchmark-memory-bandwidth' into 'main'

### Perf
- **kernel-vector-row:** new optimization: load matrix data and vector x asynchronously in pipeline

### Refactor
- **kernel-vector-row:** move imp of loading next x vector to function `load_vec_x_into_reg`
- **kernel-vector-row:** move pipeline implementation to another func vector_row_kernel_pipeline


<a name="v0.2.3"></a>
## [v0.2.3] - 2021-06-29
### Chore
- **kernel-vector-row:** add comment about row-loading in `spmv_vector_row_kernel_double_buffer`
- **typos:** fix typos in vector-row kernel: sync -> async

### Feat
- **kernel-vector-row:** basic implementation of vector-row kernel strategy for double buffer
- **tools:** better cli and sub-command "dist" to show nnz distribution
- **tools:** add a tool for dumping nnz of each row in csr matrix

### Fix
- **kernel-wf-row:** fix compiling issue of "global_mem_ops.hpp not found"

### Merge
- Merge branch 'code-refactor-and-compiling-fixes' into 'main'
- **kernel-vector-row:** Merge branch 'opt-vector-double-buffer' into 'main'
- **tools:** Merge branch 'feature-csr-tools' into 'main'

### Perf
- **kernel-vector-row:** add a new vector-row optimization: preload next row when calculating
- **kernel-vector-row:** load next data into buffer for later usage when performing calculation

### Refactor
- move enum `sparse_operation` to file src/api/types.h
- **kernel-vector-row:** remove macro `ASYNC_LOAD` in func spmv_vector_row_kernel_double_buffer
- **kernel-vector-row:** move imp of loading a row of csr data to function `load_row_into_reg`
- **kernel-vector-row:** move double buffer optimization implementation to opt_double_buffer.hpp


<a name="v0.2.2"></a>
## [v0.2.2] - 2021-06-27
### Fix
- **kernel-vector-row:** fix the condition of using LDS as buffer
- **kernel-vector-row:** fallback to use normal vector-row strategy if the LDS is not enough

### Merge
- **kernel-vector-row:** Merge branch 'fix-fallback-vector-row-when-exceed-LDS' into 'main'

### Refactor
- **kernel-vector-row:** move row calculation of a vector to a new function `vector_calc_a_row`


<a name="v0.2.1"></a>
## [v0.2.1] - 2021-06-23
### Merge
- **kernel-vector-row:** Merge branch 'opt-kernel-vector-row-memory-access' into 'main'

### Perf
- **kernel-vector-row:** change gridDim.x value (total blocks on GPU) to 512 to improve performance
- **kernel-vector-row:** use global_load_dwordx4/global_load_dwordx2 to load 2 double/int to LDS
- **kernel-vector-row:** also load column index data to LDS to achieve memory accessing optimization
- **kernel-vector-row:** memory accessing optimization: load matrix rows to LDS and than use it


<a name="v0.2.0"></a>
## [v0.2.0] - 2021-06-14
### Build
- **cmake:** move cmake generated file `building_config.h` from cmake source dir to cmake binary dir

### Chore
- add git-chglog tool config and generated file CHANGELOG.md

### Docs
- correct typos in README.md file
- **run:** update runnig document due to the new input reading (from file)

### Feat
- **demo:** sync demo code for reading csr matrix from large data set
- **demo:** add csr data files for performance test
- **demo:** update upstream demo code for reading CSR matrix from file
- **examples:** echo job node list before running jobs
- **examples:** add new batch files for using large data set (run all cases and run Hardesty3 case)
- **examples:** update sbatch script to run all test cases

### Fix
- **cmake:** fix cmake error "unsupported kernel strategy `vector_row`" when using the strategy
- **demo:** sync demo to fix wrong rows number when reading csr matrix in demo code
- **kernel-light:** correct variable names in light strategy implementation to make compiling passed

### Merge
- **cmake:** Merge branch 'fix-cmake-config-unsupported-kernel-strategy'
- **demo:** Merge branch 'demo-read-csr-file' into 'main'
- **kernel-grpup-row:** Merge branch 'feature-kernel-wf-group-row' into 'main'
- **kernel-vector-row:** Merge branch 'opt-kernel-vector-row'
- **kernel-wf-row:** Merge branch 'remove-i32-to-u64-conversion' into 'main'

### Perf
- **kernel-group-row:** rm useless `rowptr[0]` when computing average non-zeros (it is is always 0)
- **kernel-vector-row:** apply `fma` instruction to kernel-vector-row strategy
- **kernel-wf-row:** rm unnecessary `v_ashrrev_i32_e32` while converting col index (int) to address

### Refactor
- move file utils.h to common directory
- **kernel-group-row:** implementaion refactor of `group-row` and `light` strategies
- **kernel-group-row:** add two new kernel strategies, "light" and "group_row", for calculating multiple rows per wf
- **kernel-grpup-row:** variables renaming of group-row and light imp to satisfy the code style
- **kernel-grpup-row:** remove macros in group-row and light kernel strategies
- **kernel-vector-row:** rename kernel strategy `group_row` to `vector_row`

### Style
- **kernel-grpup-row:** add comments, resort headers for group-row and light kernel strategies imp


<a name="v0.1.1"></a>
## [v0.1.1] - 2021-06-03
### Perf
- **kernel-wf-row:** use shift operator, rather than multiplication and division
- **kernel-wf-row:** use global_load_dwordx4 to load 2 double in inner loop to reduce mem access

### Refactor
- **kernel-wf-row:** apply `const` to variables in function device_spmv_wf_row_default


<a name="v0.1.0"></a>
## v0.1.0 - 2021-06-03
### Build
- **cmake:** add path config of kernel source files directory for different kernel strategy
- **cmake:** add cmake building script

### Chore
- add clang-format config file
- add .gitignore file
- **examples:** add sbatch script for running the program with GPU support
- **kernel-thread-row:** remove unused file src/acc/hip-thread-row/CMakeLists.txt
- **kernel-wf-row:** remove expired comments (about `KERNEL_STRATEGY` option) in config.cmake file

### Docs
- **build:** add building document for GPU side and CPU side
- **develop:** add document for creating and using a kernel strategy
- **verify:** add building document for device/CPU side verification

### Feat
- wrapper file `acc/hip/spmv_hip_acc_imp.h` in file `Csrsparse.hpp`
- **compiling:** make compiling passed if cmake option `HIP_ENABLE_FLAG` is OFF
- **demo:** add official demo code
- **demo:** syny latest official demo code: replace clock counter `std::clock` with `gettimeofday`
- **kernel-block-row-ordinary:** one block computes one row with ordinary method
- **kernel-row-wf:** kernel strategy 'row-wf' implementation for processing one row in a wavefront
- **kernel-sync-wf-row:** kernel strategy sync-wf-row implementation for processing one row in a wavafront with synchronous thread
- **kernel-thread-row:** new strategy: one thread computes a row of A
- **kernel-wf-row-reg:** kernel strategy wf-row-reg for processing one row in a wavafront with register __shfl_down

### Fix
- **compiling:** make compiling and validation passed
- **demo:** sync the latest demo code, which fixed a bug of matrix A generation: "A is always dense"
- **demo:** fix building error `use of undeclared identifier 'cout'` due to missing `iostream` header
- **macro:** add missing macro `gpu` generated by cmake `configure_file`
- **reg-reduce:** correct the incorrect use of the __shfl_down function to fix failed results validation

### Merge
- Merge branch 'refactor-merge-wf-row' into 'main'
- Merge branch 'sync-wf-row' into refactor-merge-wf-row
- Merge branch 'make-compiling-passed' into 'main'
- Merge branch 'feature-kernel-wf-row-reg' into 'main'
- Merge branch 'feature-strategy-row-wf' into 'main'
- Merge branch 'feature-kernel-strategy' into 'main'
- **cmake:** Merge branch 'cmake-configs' into 'main'
- **demo:** Merge branch 'official-demo-code' into 'main'
- **kernel-block-row-ordinary:** Merge branch 'feature-kernel-block-row-ordinary' into 'main'
- **kernel-thread-row:** Merge branch 'feature-strategy-thread-row' into 'main'
- **kernel-wf-row:** Merge branch 'opt-wf-row-strategy' into 'main'
- **kernel-wf-row:** Merge branch 'opt-row-wf-strategy' into main

### Perf
- **kernel-wf-row:** move `alpha` multiplication out of the inner col-loop
- **kernel-wf-row:** adjust kernel threads number and block number to make full use of CUs of GPU
- **kernel-wf-row:** adjust threads number in a block to make full use of SIMDs in GPU

### Refactor
- **kernel-thread-row:** code refactor that may improve performance: move variables out of loop
- **kernel-wf-row:** merge 3 wavefront reduce methods(default,LDS,Reg) all into wf-row strategy
- **kernel-wf-row:** rename strategy 'row-wf' to 'wf-row'

### Style
- **demo:** code format of official demo code
- **kernel-block-row-ordinary:** code format for implementation code of block-row-ordinary strategy
- **kernel-thread-row:** code format of implementation of thread-row strategy
- **kernel-wf-row-reg:** removed unused code


[Unreleased]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.3.0...HEAD
[v0.3.0]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.4...v0.3.0
[v0.2.4]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.3...v0.2.4
[v0.2.3]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.2...v0.2.3
[v0.2.2]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.1...v0.2.2
[v0.2.1]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.0...v0.2.1
[v0.2.0]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.1.1...v0.2.0
[v0.1.1]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.1.0...v0.1.1
