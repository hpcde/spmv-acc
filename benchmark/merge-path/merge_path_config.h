#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_CONFIG_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_CONFIG_H

enum ReductionAlgorithm { Linear = 0, Binary = 1 };

struct LinearSearchType {};
struct BinarySearchType {};

template <int ReductionAlgorithm> struct ReductionTrait;

template <> struct ReductionTrait<Linear> { using type = LinearSearchType; };

template <> struct ReductionTrait<Binary> { using type = BinarySearchType; };

enum UpdateAlgorithm { SingleBlock = 0, LookBack = 1 };

struct SingleBlockType {};
struct LookBackType {};

template <int UpdateAlgorithm> struct UpdateTrait;

template <> struct UpdateTrait<SingleBlock> { using type = SingleBlockType; };

template <> struct UpdateTrait<LookBack> { using type = LookBackType; };

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_CONFIG_H