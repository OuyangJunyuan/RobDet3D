#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

#define THREAD_SIZE 256
#define BLOCKS1D(M) dim3(((M)+THREAD_SIZE-1)/THREAD_SIZE)
#define BLOCKS2D(M, B) dim3((((M)+THREAD_SIZE-1)/THREAD_SIZE),B)
#define THREADS() dim3(THREAD_SIZE)

#define PTR(x, t) (x.data_ptr<t>())
#define DECL_PTR(X, T1, T2) auto X##_ptr = (T1*)PTR(X,T2)
#define AT_TYPE(X, TYPE) (at::device(X.device()).dtype(at::ScalarType::TYPE))

constexpr uint32_t kEmpty = 0xffffffff;

inline auto get_table_size(int64_t N, int64_t min_size) {
    int64_t table_size = std::max(min_size, N * 2);
    table_size = (2 << ((uint64_t) ceil((log((double) table_size) / log(2.0))) - 1));
    return table_size;
}

__device__ __forceinline__
float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__
float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__
float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__
void atomicMin(float *address, float val) {
    int *address_as_i = (int *) address;
    int old = *address_as_i;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(min(val, __int_as_float(assumed))));
    } while (assumed != old);  // fail to insert the min val since *address_as_i was changed before atomicCAS execution.
//    return __int_as_float(old);
}

__device__ __forceinline__
uint32_t coord_hash_32(const int x, const int y, const int z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}

__device__ __forceinline__
uint32_t coord_hash_32(const int &b, const int &x, const int &y, const int &z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (b + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}

__global__
void voxel_update_kernel(const uint32_t batch_size, const uint32_t sample_num, const float threshold,
                         const uint32_t *__restrict__ sampled_num, bool *__restrict__ batch_mask,
                         float3 *__restrict__ voxel_cur, float3 *__restrict__ voxel_lower,
                         float3 *__restrict__ voxel_upper);
