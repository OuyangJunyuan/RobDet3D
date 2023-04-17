#include <cuda.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"


__global__
void valid_voxel_kernel(const uint32_t N, const uint32_t MAX, const bool *__restrict__ mask,
                        const uint64_t *__restrict__ bid, const float3 *__restrict__ xyz,
                        const float3 *__restrict__ voxel, uint32_t *__restrict__ table,
                        uint32_t *__restrict__ count) {
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (pts_id >= N || mask[bid[pts_id]]) return;
    auto batch_id = bid[pts_id];

    auto v = voxel[batch_id], pt = xyz[pts_id];
    uint32_t key = coord_hash_32(batch_id, roundf(pt.x / v.x), roundf(pt.y / v.y), roundf(pt.z / v.z));
    uint32_t slot = key & MAX;
    while (true) {
        uint32_t prev = atomicCAS(table + slot, kEmpty, key);
        if (prev == key) { return; }
        if (prev == kEmpty) {
            atomicAdd(count + batch_id, 1);
            return;
        }
        slot = (slot + 1) & MAX;
    }
}


__global__
void unique_mini_dist_kernel(const uint32_t N, const uint32_t MAX,
                             const uint64_t *__restrict__ bid, const float3 *__restrict__ xyz,
                             const float3 *__restrict__ voxel,
                             uint32_t *__restrict__ key_table, float *__restrict__ dist_table,
                             uint32_t *__restrict__ pts_slot, float *__restrict__ pts_dist) {

    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (pts_id >= N) return;

    auto batch_id = bid[pts_id];
    auto pt = xyz[pts_id], v = voxel[batch_id];
    auto coord_x = roundf(pt.x / v.x);
    auto coord_y = roundf(pt.y / v.y);
    auto coord_z = roundf(pt.z / v.z);
    auto d1 = pt.x - coord_x * v.x;
    auto d2 = pt.y - coord_y * v.y;
    auto d3 = pt.z - coord_z * v.z;
    auto dist = d1 * d1 + d2 * d2 + d3 * d3;

    // insert to table.
    uint32_t key = coord_hash_32(batch_id, coord_x, coord_y, coord_z);
    uint32_t slot = key & MAX;
    while (true) {
        uint32_t prev = atomicCAS(key_table + slot, kEmpty, key);
        if (prev == key or prev == kEmpty) {  // the value of this slot in hash maybe reset by last iteration
            atomicMin(dist_table + slot, dist);
            pts_slot[pts_id] = slot;
            pts_dist[pts_id] = dist;
            return;
        }
        slot = (slot + 1) & MAX;
    }
}


__global__
void set_mask_kernel(const uint32_t N, const uint32_t S,
                     const uint64_t *__restrict__ pts_bid,
                     const uint32_t *__restrict__ pts_slot,
                     const float *__restrict__ pts_dist,
                     float *__restrict__ table,
                     uint64_t *__restrict__ ind,
                     uint32_t *__restrict__ count) {
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (pts_id >= N) return;

    auto batch_id = pts_bid[pts_id];
    auto *min_dist = table + pts_slot[pts_id];
    if (*min_dist == pts_dist[pts_id]) {
        auto cnt = atomicAdd(count + batch_id, 1);
        if (cnt < S) {
            atomicExch(min_dist, FLT_MAX);
            ind[batch_id * S + cnt] = pts_id;
        }
    }
}


/***
 * @param point_bid point batch id in long(N1 + ... + Nb, 1)
 * @param point_xyz point coords in float(N1 + ... + Nb, 3)
 * @param batch_size the max number in point_bid
 * @param sample_num the number of points to be sampled in each batch
 * @param voxel initial voxel size, a potential voxel size will be searched in range of 0 ~ 2voxel
 * @param threshold the upper tolerance of sample_num
 * @param max_itr the allowed maximum iteration times
 * @return sampled_ind
 */
at::Tensor potential_voxel_size_stack_wrapper(const at::Tensor &point_bid,
                                              const at::Tensor &point_xyz,
                                              const int &batch_size,
                                              const int &sample_num,
                                              const std::vector<float> &voxel,
                                              const float &threshold,
                                              const int &max_itr) {
    uint32_t
            S = sample_num,
            B = batch_size,
            N = point_xyz.size(0),
            T = get_table_size(N, 2048),
            MAX = T - 1;

    auto
            init_voxel = torch::tensor({{voxel[0], voxel[1], voxel[2]}}, AT_TYPE(point_xyz, Float)),
            voxel_lower = torch::zeros({B, 3}, AT_TYPE(point_xyz, Float)),
            voxel_cur = init_voxel.repeat({B, 1}).contiguous(),
            voxel_upper = voxel_cur * 2.0,

            batch_mask = torch::zeros({B}, AT_TYPE(point_xyz, Bool)),
            hash_table = torch::empty({T}, AT_TYPE(point_xyz, Int)),
            dist_table = torch::full({T}, at::Scalar(FLT_MAX), AT_TYPE(point_xyz, Float)),
            point_slot = torch::empty({N}, AT_TYPE(point_xyz, Int)),
            point_dist = torch::empty({N}, AT_TYPE(point_xyz, Float)),
            sampled_num = torch::empty({B}, AT_TYPE(point_xyz, Int)),
            sampled_ind = torch::zeros({B * S}, AT_TYPE(point_xyz, Long));


    auto iter_num = 0;
    auto hash_empty = at::Scalar((int) 0xffffffff);
    auto blocks = BLOCKS1D(N), threads = THREADS();
    DECL_PTR(voxel_cur, float3, float);  // current voxel size.
    DECL_PTR(voxel_lower, float3, float);  // the lower bound of voxel size to be searched.
    DECL_PTR(voxel_upper, float3, float);  // the upper bound of voxel size to be searched.
    DECL_PTR(batch_mask, bool, bool);  // is this batch sampled completely?
    DECL_PTR(hash_table, uint32_t, int);  // the hash table for voxel coord inserting.
    DECL_PTR(dist_table, float, float);  // the table to store the minimum distance.
    DECL_PTR(point_bid, uint64_t, long);  // the batch id of each point.
    DECL_PTR(point_xyz, float3, float);  // the coordination of each point.
    DECL_PTR(point_slot, uint32_t, int);  // the position in hash-table for each point.
    DECL_PTR(point_dist, float, float);  // the minimum distance from each point to its corresponding voxel center.
    DECL_PTR(sampled_ind, uint64_t, long); // the output sampled stacked indices.
    DECL_PTR(sampled_num, uint32_t, int);  // how many points have been sampled in each batch.

    // compute potential voxel size.
    do {
        sampled_num.zero_();
        hash_table.fill_(hash_empty);
        valid_voxel_kernel<<< blocks, threads>>>(
                N, MAX,
                batch_mask_ptr,
                point_bid_ptr,
                point_xyz_ptr,
                voxel_cur_ptr,
                hash_table_ptr,
                sampled_num_ptr
        );
        voxel_update_kernel<<<1, B>>>(
                B, S, threshold,
                sampled_num_ptr,
                batch_mask_ptr,
                voxel_cur_ptr,
                voxel_lower_ptr,
                voxel_upper_ptr
        );
    } while (++iter_num <= max_itr and not batch_mask.all().item().toBool());

    // compute mask that indicate if each point is the closet one to its corresponding voxel center.
    sampled_num.zero_();
    unique_mini_dist_kernel<<< blocks, threads>>>(
            N, MAX,
            point_bid_ptr,
            point_xyz_ptr,
            voxel_cur_ptr,
            hash_table_ptr,
            dist_table_ptr,
            point_slot_ptr,
            point_dist_ptr
    );
    set_mask_kernel<<< blocks, threads>>>(
            N, S,
            point_bid_ptr,
            point_slot_ptr,
            point_dist_ptr,
            dist_table_ptr,
            sampled_ind_ptr,
            sampled_num_ptr
    );
    return sampled_ind;
}

