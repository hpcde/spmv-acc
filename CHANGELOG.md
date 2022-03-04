<a name="unreleased"></a>
## [Unreleased]


<a name="v0.6.0"></a>
## [v0.6.0] - 2022-03-01
### Build
- **cmake:** support OpenMP for benchmark executable target when performing linking
- **cmake:** reorganize benchmark device-side libs and pass the missing HIP_NVCC_FLAGS to flat src
- **cmake:** add a cmake option `SPMV_BUILD_BENCHMARK` to disable/enable benchmark building
- **cmake:** reorganize cmake script under directory benchmark
- **cmake:** add cmake config for benchmark executable binary
- **cmake:** set CXX standard to 14 for using std::unique_ptr
- **cmake:** move clipp.h header to directory `third-party`

### Docs
- **readme:** add document of fetching dependency `clipp`
- **suitesparse-dl:** add document for "matrices with the same name"
- **suitesparse-dl:** add document of extracting matrix market from .tar.gz file

### Feat
- correctly access row_offset when device row_offset accessing on host side is not supported
- **benchmark:** add benchmark support for SpMV-acc strategies and cuSPARSE in func `test_spmv`
- **benchmark:** more strong verification for benchmark
- **benchmark:** throw runtime_error of hola device SpMV
- **benchmark:** record preproc time, calc time and destroy time in statistics data
- **benchmark:** add the adaptive line kernel to benchmark
- **benchmark:** record strategy name in statistics data
- **benchmark:** benchmark support for rocsparse SpMV: vector and csr-adaptive
- **benchmark:** benchmark support for hola CsrMV on hip
- **benchmark:** benchmark support for hola CsrMV on Cuda
- **benchmark:** destory host/device data after test
- **benchmark:** disable benchmarking the default sequential spmv method on device side
- **benchmark:** verify after outputting statistics
- **benchmark:** change style of performance statistics log: split header and data
- **benchmark:** benchmark support for cub CsrMV on Cuda
- **cli:** add OpenMP support to matrix-market input parsing
- **cli:** update the cli and benchmark help message: value of `-f` can be `bin`
- **cli:** add OpenMP support for data sorting when converting COO to CSR
- **cli:** support to read csr-binary format converted by `suitesparse-dl conv` sub-command
- **cli:** use clipp to parse command line argument
- **cli:** show more detailed error message in HIP_CHECK macro
- **cli:** add matrix market format reading support
- **common:** support reduction on CUDA platform and make DPP reduction available only on ROCm
- **common:** add compatible code to make nontemporal-load/store, fma and atomicAdd work on nvidia
- **common:** mv data-load assembly to `platform/rocm` and add compatible data loading code for CUDA
- **kernel-light:** support the case of "wrap size is 32" (Nvidia GPU) for light strategy
- **kernel-line:** replace the origin line method with adaptive-line method
- **kernel-line:** adjust the value of ROW_NUM in adaptive line calculation
- **kernel-line:** add adaptive line-kerenel: switch between line and vector-row
- **kernel-line:** combine the LDS space used by line and vector-row algorithm in adaptive line
- **kernel-line-enhance:** support memory coalescing in vector reduction when storing data back
- **kernel-line-enhance:** add a static assert to check vector number and rows assigned to a block
- **kernel-vector-row:** support to move data from thread 0 of vectors to front lanes on CUDA
- **kernel-vertor-row:** support the case of "wrap size is 32" (Nvidia GPU)
- **scripts:** add a compiler wrapper to remove unrecognized argument `-std=gnu++14` for nvcc
- **suitesparse-dl:** download matrices into different NNZ categories
- **suitesparse-dl:** support to convert matrix market format to binary CSR file format
- **suitesparse-dl:** add feature of generating sbatch file from matrices data for job submitting
- **suitesparse-dl:** add `fetch` subcommand for fetching collection metadata
- **suitesparse-dl:** add cli flags parsing and use `dl` sub-command for downloading
- **suitesparse-dl:** add cli flags to specific data dir and goroutines for downloading
- **suitesparse-dl:** support to convert matrix with integer type as value
- **suitesparse-dl:** stop downloading if there is any error in downloading process
- **suitesparse-dl:** skip the matrix downloading if the file exists
- **suitesparse-dl:** download matrix to temporary file and then rename to the final file
- **suitesparse-dl:** add a tool for downloading matrices of matrix market format from SuiteSparse
- **third-party:** add a prompt to set the WARP_SIZE after downloaded hip-hola
- **third-party:** specific version and apply changes when or after downloading hola

### Fix
- **benchmark:** correct verification for cub and hola spmv which only compute y=Ax
- **benchmark:** fix cub SpMV invalid device function
- **cli:** fix incorrect nnz assertion while reading matrix market format
- **cli:** fix Segmentation fault while using csr matrix data after variable `csr_reader` is released
- **cmake:** fix typos and incorrect path in benchmark cmake script to make hola-hip compiling passed
- **compile:** correct possible compiling error `error: size of array 'buf' is negative`
- **compile:** fix possible compiling error of `determining template type` under some compilers
- **compile:** fix compiling errors in benchmark and cli building
- **compile:** fix compiling issues (with OpenMP support) on nvidia platform
- **compile:** add the missing template param for func `dpp_wf_reduce_sum` on ROCm platform
- **demo:** fix possible crash when allocing buffer memory for csr file reading in demo
- **kernel-flat:** fix compiling error: `flat_sparse_spmv` declaration does not match the definition
- **kernel-line:** fix incorrect data storing logic of vector-row method in adaptive line
- **kernel-line-enhance:** fix the possible stucked vector-based reduction on NVIDIA platform
- **suitesparse-dl:** fix incorrect "NNZ" values stored into binary csr file header
- **suitesparse-dl:** fix compiling errors in `fetch` sub-command
- **suitesparse-dl:** fix parsing error when the matrix data type is complex
- **third-party:** bump hola-hip version to fix building errors from `hipFree` and `CSR<double>` template

### Merge
- Merge pull request [#35](https://github.com/hpcde/spmv-acc/issues/35) from hpcde/fix-hola-hip-building-errors
- Merge pull request [#25](https://github.com/hpcde/spmv-acc/issues/25) from hpcde/feature-nvidia-support
- Merge pull request [#27](https://github.com/hpcde/spmv-acc/issues/27) from hpcde/fix-error-of-nvidia-support
- **benchmark:** Merge pull request [#28](https://github.com/hpcde/spmv-acc/issues/28) from hpcde/feature-benchmark
- **benchmark:** Merge pull request [#36](https://github.com/hpcde/spmv-acc/issues/36) from hpcde/enhanced-HIP_CHECK-and-verify-for-cli-and-benchmark
- **benchmark:** Merge pull request [#30](https://github.com/hpcde/spmv-acc/issues/30) from hpcde/benchmarks-improves
- **cli:** Merge pull request [#26](https://github.com/hpcde/spmv-acc/issues/26) from hpcde/feature-openmp-matrix-market-reading
- **cli:** Merge pull request [#31](https://github.com/hpcde/spmv-acc/issues/31) from hpcde/feature-csr-binary-reader
- **cli:** Merge pull request [#22](https://github.com/hpcde/spmv-acc/issues/22) from hpcde/feature-matrix_market_support
- **kernel-line:** Merge pull request [#32](https://github.com/hpcde/spmv-acc/issues/32) from hpcde/feature-adaptive-line
- **kernel-line-enhance:** Merge pull request [#34](https://github.com/hpcde/spmv-acc/issues/34) from hpcde/fix-line-enhance-shlf_down-reduce-stucked
- **kernel-line-enhance:** Merge pull request [#37](https://github.com/hpcde/spmv-acc/issues/37) from hpcde/opt-line-enhance-vector-reduction
- **suitesparse-dl:** Merge pull request [#33](https://github.com/hpcde/spmv-acc/issues/33) from hpcde/hotfix-suitesparse-dl-conv
- **suitesparse-dl:** Merge pull request [#24](https://github.com/hpcde/spmv-acc/issues/24) from hpcde/feature-suitesparse-dl-sbatch-gen
- **suitesparse-dl:** merge: Merge pull request [#23](https://github.com/hpcde/spmv-acc/issues/23) from hpcde/feature-suitesparse-downloader

### Perf
- **cli:** remove unnecessary data copy when converting matrix-market to CSR
- **cli:** replace stringstream with user-customized line parser to parse matrix-market format
- **cli:** read the whole file into buffer and than parse while loading matrix-market format
- **kernel-line-enhance:** move a part of sum reduction (local & global shift) to out of rounds-loop

### Refactor
- **benchmark:** apply a simpler approach to benchmark different algorithms
- **cli:** refactor code of matrix-market reading and parsing: split header and body reading
- **cli:** move implementation code of verification.h to cpp file
- **cli:** rename class mtx_reader -> csr_mtx_reader, file csr.hpp -> sparse_format.h
- **kernel-line:** use param BLOCK_LDS_SIZE, rather than MAX_ROW_NNZ, to specific LDS size

### Pull Requests
- Merge pull request [#29](https://github.com/hpcde/spmv-acc/issues/29) from hpcde/feature-suitesparse-dl-binary-csr-convert


<a name="v0.5.0"></a>
## [v0.5.0] - 2021-10-15
### Chore
- add LICENSE file

### Docs
- **changelog:** update changelog for v0.5.0
- **changelog:** update changelog
- **readme:** add citing information

### Feat
- **kernel-adaptive:** use adaptive line-enhance when the matrix is small or is short-row matrix
- **kernel-adaptive:** apply line strategy(one-pass line) to short-row matrices in adaptive strategy
- **kernel-line-enhance:** add adaptive line_enhance, which set kernel template parameters adaptively

### Merge
- **kernel-adaptive:** Merge branch 'revert-adaptive-line-enhance-case' into branch main
- **kernel-adaptive:** Merge pull request [#20](https://git.hpcer.dev/PRA/spmv-acc/issues/20) from hpcde/feature-adaptive-line-enhance
- **kernel-adaptive:** Merge pull request [#19](https://git.hpcer.dev/PRA/spmv-acc/issues/19) from hpcde/feature-kernel-line-for-short-row-matrix

### Revert
- **kernel-adaptive:** don't use adaptive line-enhance if a large matrix has short rows


<a name="stash_flat_preprocess"></a>
## [stash_flat_preprocess] - 2021-10-07
### Build
- **cmake:** fix cmake find OpenMP errors while OpenMP target mode is not support
- **cmake:** change the files (only headers in src/acc dir) and lib (kernel lib) for installation

### Docs
- **readme:** change kernel strategy in example build script, and add docs of min rocsparse version

### Feat
- **cli:** refactor the cli part of reading file, calling spmv, verifing, and put source to cli dir
- **cmake:** set default value of cmake option `SPMV_OMP_ENABLED_FLAG` to `OFF`
- **kernel-adaptive:** change condition of using `flat` strategy to "matrix nnz is larger than 2^23"
- **kernel-flat:** add a new vector based reduction method for flat kernel strategy
- **kernel-flat:** support the case "vec number is larger than reduce rows" in vec-level reduction
- **kernel-flat:** add a new vector based reduction method with memory coalescing
- **kernel-flat:** support to set flat kernel template parameters adaptively using average row nnz
- **kernel-flat:** use config `FLAT_ONE_PASS_ADAPTIVE` to enable/disable adaptive flat
- **kernel-flat:** apply adaptive flat to kernel strategy `FLAT`
- **kernel-line-enhance:** create kernel strategy `line-enhance` with empty implementation
- **kernel-line-enhance:** add kernel func implementation of strategy `line-enhance`
- **tool:** use muellan/clipp lib for cli argument parsing
- **tools:** feature of showing matrix nnz by part or row

### Fix
- **kernel-flat:** correct the wrong length of array `break_points`

### Merge
- Merge pull request [#14](https://git.hpcer.dev/PRA/spmv-acc/issues/14) from hpcde/feature-more-flexible-configs
- **cli:** Merge pull request [#11](https://git.hpcer.dev/PRA/spmv-acc/issues/11) from hpcde/feature-new-cli
- **cmake:** Merge pull request [#12](https://git.hpcer.dev/PRA/spmv-acc/issues/12) from hpcde/fix-cmake-find-omp
- **kernel-flat:** Merge pull request [#10](https://git.hpcer.dev/PRA/spmv-acc/issues/10) from hpcde/feature-kernel-flat-vec-reduction
- **kernel-flat:** Merge pull request [#13](https://git.hpcer.dev/PRA/spmv-acc/issues/13) from hpcde/feature-kernel-flat-one-pass
- **kernel-line:** Merge pull request [#15](https://git.hpcer.dev/PRA/spmv-acc/issues/15) from hpcde/feature-kernel-line-one-pass
- **kernel-line-enhance:** Merge pull request [#18](https://git.hpcer.dev/PRA/spmv-acc/issues/18) from hpcde/feature-kernel-strategy-line-enhance
- **tools:** Merge pull request [#16](https://git.hpcer.dev/PRA/spmv-acc/issues/16) from hpcde/feature-csr-tool-block-nnz

### Perf
- **cli:** add OpenMP support for the new cli to parse the input matrix
- **kernel-flat:** add another flat imp: each block only iterate one pass in its lifetime
- **kernel-line:** set default kernel threads number (`HIP_THREAD`) to `256` for line-one-pass
- **kernel-line:** add optimization of loading 2 matrix values each time in line-one-pass
- **kernel-line:** add `R` param, which can assign each block `R*blockDim.x/row_max_nnz` rows
- **kernel-line:** add another line imp: each block only iterate one pass in its lifetime
- **kernel-line-enhance:** new reduce method: divide block into vectors and use vector for reduction

### Refactor
- use wavefront size generated by cmake configure (config by cmake option `WAVEFRONT_SIZE`)
- **kernel-flat:** instantiate flat parameters (reduce method, vec size and R) in func template
- **kernel-flat:** move all reduction kernel functions to file `flat_reduce.hpp`
- **kernel-flat:** move direct reduction code in flat strategy to new func `flat_reduce_direct`
- **kernel-line-enhance:** move direct reduction to a new func `line_enhance_direct_reduce`

### Pull Requests
- Merge pull request [#17](https://git.hpcer.dev/PRA/spmv-acc/issues/17) from hpcde/feature-adaptive-flat


<a name="v0.4.0"></a>
## [v0.4.0] - 2021-08-31
### Build
- **cmake:** gen strategy config from cmake and apply strategy via the macro in generated header
- **cmake:** enable to build all strategies code via cmake script

### Chore
- **changelog:** update CHANGELOG.md for v0.3.0
- **examples:** reorder the case in alphabetical order and correct cases information
- **kernel-thread-row:** improved code comments of kernel function `kernel_thread_row_v2`

### Docs
- **changelog:** update change log for v0.4.0
- **develop:** update document of adding a new kernel strategy in README.md

### Feat
- **kernel-adaptive:** add config for the new kernel strategy `adaptive`
- **kernel-adaptive:** apply matrix data dividing based vector-row if it is available
- **kernel-adaptive:** add basic implementation of adaptive method (no data blocks dividing)
- **kernel-thread-row:** remove useless `__syncthreads` in kernel func `kernel_thread_row_block_v2`
- **kernel-thread-row:** call kernel func `kernel_thread_row_block_v2`(x remapping at block level)
- **kernel-thread-row:** apply vector x reampping when macro `OPT_THREAD_ROW_REMAP_VEC_X` is defined
- **kernel-thread-row:** add another block level imp: one thread only process a row in its lifetime
- **kernel-thread-row:** add another imp of "thread-row" strategy with calculation in block level
- **kernel-vector-row:** adaptive kernel: apply different `VECTOR_SIZE` to data blocks of matrix A

### Fix
- **kernel-flat:** fix incorrect results when reduction size is larger than threads number in block
- **kernel-thread-row:** correct the limit of "template param N can only be 1" at block level
- **kernel-vector-row:** fix the wrong kernel func called in native vector-row while VECTOR_SIZE is 8

### Merge
- Merge pull request [#4](https://git.hpcer.dev/PRA/spmv-acc/issues/4) from hpcde/fix-flat-and-vector-row-bugs
- Merge branch 'cmake-enable-build-all-strategies' into 'main'
- **kernel-adaptive:** Merge pull request [#1](https://git.hpcer.dev/PRA/spmv-acc/issues/1) from hpcde/feature-kernel-adaptive into branch main
- **kernel-flat:** Merge pull request [#2](https://git.hpcer.dev/PRA/spmv-acc/issues/2) from hpcde/opt-on-kernel-flat
- **kernel-thread-row:** Merge pull request [#9](https://git.hpcer.dev/PRA/spmv-acc/issues/9) from hpcde/feat-thread-row-single
- **kernel-thread-row:** Merge pull request [#6](https://git.hpcer.dev/PRA/spmv-acc/issues/6) from hpcde/opt-thread-row-in-block-level
- **kernel-thread-row:** Merge pull request [#8](https://git.hpcer.dev/PRA/spmv-acc/issues/8) from hpcde/opt-thread-row-tune-kernel-config
- **kernel-thread-row:** Merge pull request [#7](https://git.hpcer.dev/PRA/spmv-acc/issues/7) from hpcde/opt-thread-row-in-block-level-x-remap
- **kernel-vector-row:** Merge branch 'enhance-vector-row' into feature-kernel-adaptive
- **kernel-vector-row:** Merge pull request [#3](https://git.hpcer.dev/PRA/spmv-acc/issues/3) from hpcde/opt-vector-row-access-y-coalescing
- **kernel-vector-row:** Merge branch 'enhance-vector-row' into 'main'
- **thread-row:** Merge pull request [#5](https://git.hpcer.dev/PRA/spmv-acc/issues/5) from hpcde/opt-thread-row-remap-vec_x

### Perf
- **kernel-flat:** move barrier __syncthreads() ahead(moved to the place before loading matrix data)
- **kernel-flat:** apply loop unrolling to data loading of csr matrix and x vector
- **kernel-thread-row:** tune the BLOCK dim of thread-row at block level(with x remapping) to 512
- **kernel-thread-row:** use LDS and __shfl to load start/end row index of block to threads in block
- **kernel-thread-row:** make the memory accessing mode of $x$ as column-first mode at block level
- **kernel-thread-row:** apply global_load_dwordx4/x2 to load 2 double/int to LDS at block level
- **kernel-thread-row:** tune grid dim for native thread-row and optimization "vec x remapping"
- **kernel-thread-row:** break the row-loop if the start row index is larger/equal than matrix rows
- **kernel-thread-row:** use native C++ code, instead of asssembly, to load 2 matrix values into LDS
- **kernel-thread-row:** tune kernel config of thread-row at block level to <<<7010, 512>>>
- **kernel-thread-row:** add a new kernel func for remapping memory access pattern of vector $x$
- **kernel-vector-row:** memory coalescing of loading and storing y from vectors
- **kernel-vector-row:** memory access coalescing at block level when loading and storing vector y
- **kernel-vector-row:** add vector y memory coalescing support for adaptive vector-row strategy
- **kernel-vector-row:** simple load-balance of wavefronts number on data blocks

### Refactor
- functions renaming to solve signature conflict when building all strategies into one lib
- **kernel-flat:** use template param `THRAEDS`, instead of blockDim.x, as threads num in block
- **kernel-thread-row:** refactor macro control on different optimization kernels in thread-row
- **kernel-thread-row:** move kernel func `kernel_thread_row_v2` to file thread_row_x_remap.inl
- **kernel-thread-row:** mv kernel native_thread_row to native_thread_row.cpp to fix build error
- **kernel-thread-row:** use constexpr to replace C macro for thread-row optimization selection
- **kernel-vector-row:** extract the imp of y memory coalescing to `store_y_with_coalescing`

### Revert
- **kernel-vector-row:** recover the missing native vector-row method

### Style
- **kernel-thread-row:** code format of file `thread_row_block_x_remap.hpp`


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


[Unreleased]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.5.0...HEAD
[v0.5.0]: https://git.hpcer.dev/PRA/spmv-acc/compare/stash_flat_preprocess...v0.5.0
[stash_flat_preprocess]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.4.0...stash_flat_preprocess
[v0.4.0]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.3.0...v0.4.0
[v0.3.0]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.4...v0.3.0
[v0.2.4]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.3...v0.2.4
[v0.2.3]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.2...v0.2.3
[v0.2.2]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.1...v0.2.2
[v0.2.1]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.2.0...v0.2.1
[v0.2.0]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.1.1...v0.2.0
[v0.1.1]: https://git.hpcer.dev/PRA/spmv-acc/compare/v0.1.0...v0.1.1
