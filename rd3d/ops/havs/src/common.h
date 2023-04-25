#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

#ifdef DEBUG
#define dbg(STR, ...) printf(#__VA_ARGS__ ": " STR "\n" ,__VA_ARGS__)
#else
#define dbg(...)
#endif

#define THREAD_SIZE 256
#define BLOCKS1D(M) dim3(((M)+THREAD_SIZE-1)/THREAD_SIZE)
#define BLOCKS2D(M, B) dim3((((M)+THREAD_SIZE-1)/THREAD_SIZE),B)
#define THREADS() dim3(THREAD_SIZE)

#define PTR(x, t) (x.data_ptr<t>())
#define DECL_PTR(X, T1, T2) auto X##_ptr = (T1*)PTR(X,T2)
#define AT_TYPE(X, TYPE) (at::device(X.device()).dtype(at::ScalarType::TYPE))

constexpr uint32_t kEmpty = 0xffffffff;


template<typename T>
constexpr auto AsTorchType() {
    static_assert(sizeof(T) == 0, "unsupported");
    return at::ScalarType::Undefined;
}

template<>
constexpr auto AsTorchType<bool>() {
    return at::ScalarType::Bool;
}

template<>
constexpr auto AsTorchType<float>() {
    return at::ScalarType::Float;
}

template<>
constexpr auto AsTorchType<double>() {
    return at::ScalarType::Double;
}

template<>
constexpr auto AsTorchType<char>() {
    return at::ScalarType::Char;
}

template<>
constexpr auto AsTorchType<unsigned char>() {
    return at::ScalarType::Byte;
}

template<>
constexpr auto AsTorchType<short>() {
    return at::ScalarType::Short;
}

template<>
constexpr auto AsTorchType<int>() {
    return at::ScalarType::Int;
}

template<>
constexpr auto AsTorchType<long>() {
    return at::ScalarType::Long;
}

template<typename T>
constexpr auto TorchType = AsTorchType<T>();

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
float3 operator+(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__
float3 operator-(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__
float3 operator/(const float3 a, const float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__
float3 operator/(const float3 a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
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
uint32_t coord_hash_32(const int b, const int x, const int y, const int z) {
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


inline auto get_table_size(int64_t nums, int64_t min = 2048) {
    auto table_size = nums * 2 > min ? nums * 2 : min;
    table_size = 2 << ((int64_t) ceilf((logf(static_cast<float>(table_size)) / logf(2.0f))) - 1);
    return table_size;
}