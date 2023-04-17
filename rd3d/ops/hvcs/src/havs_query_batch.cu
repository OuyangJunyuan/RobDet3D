#include <cuda.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"

/***
 * 30us for 2080Ti with grid:  <<<64, 8, 1>>> and block: <<<256, 1, 1>>>
 * @param B
 * @param N
 * @param T
 * @param MAX
 * @param mask
 * @param xyz
 * @param voxel
 * @param table
 * @param count
 */
__global__
void valid_voxel_batch(const uint32_t N, const uint32_t T,
                       const bool *__restrict__ mask, const float3 *__restrict__ xyz, const float3 *__restrict__ voxel,
                       uint32_t *__restrict__ table, uint32_t *__restrict__ count) {
    uint32_t batch_id = blockIdx.y;
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (mask[batch_id] || pts_id >= N) return;
    const uint32_t MAX = T - 1;  // 0x00..ff..ff

    auto pt_this_batch = N * batch_id + pts_id;
    auto table_this_batch = table + T * batch_id;
    auto pt = xyz[pt_this_batch], v = voxel[batch_id];

    uint32_t key = coord_hash_32(roundf(pt.x / v.x), roundf(pt.y / v.y), roundf(pt.z / v.z));
    uint32_t slot = key & MAX;

    while (true) {
        uint32_t prev = atomicCAS(table_this_batch + slot, kEmpty, key);
        if (prev == key) {  // this voxel had been handled.
            return;
        }
        if (prev == kEmpty) {  // the first the to handle this voxel.
            atomicAdd(count + batch_id, 1);
            return;
        }
        slot = (slot + 1) & MAX;
    }
}


__global__
void find_mini_dist_for_valid_voxels_batch(const uint32_t N, const uint32_t T,
                                           const float3 *__restrict__ xyz, const float3 *__restrict__ voxel,
                                           uint32_t *__restrict__ key_table, float *__restrict__ dist_table,
                                           uint32_t *__restrict__ pts_slot, float *__restrict__ pts_dist) {

    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (pts_id >= N) return;
    uint32_t batch_id = blockIdx.y;

    const uint32_t MAX = T - 1;
    auto pt_this_batch = N * batch_id + pts_id;
    auto key_this_batch = key_table + T * batch_id;

    auto pt = xyz[pt_this_batch], v = voxel[batch_id];

    auto coord_x = roundf(pt.x / v.x);
    auto coord_y = roundf(pt.y / v.y);
    auto coord_z = roundf(pt.z / v.z);
    auto d1 = pt.x - coord_x * v.x;
    auto d2 = pt.y - coord_y * v.y;
    auto d3 = pt.z - coord_z * v.z;

    pts_dist[pt_this_batch] = d1 * d1 + d2 * d2 + d3 * d3;
    uint32_t key = coord_hash_32(coord_x, coord_y, coord_z);
    uint32_t slot = key & MAX;

    while (true) {
        uint32_t prev = atomicCAS(key_this_batch + slot, kEmpty, key);
        if (prev == key or prev == kEmpty) {
            atomicMin(dist_table + batch_id * T + slot, pts_dist[pt_this_batch]);
            pts_slot[pt_this_batch] = slot;
            return;
        }
        slot = (slot + 1) & MAX;
    }
}

__global__
void mask_input_if_with_min_dist_batch(const uint32_t N, const uint32_t T,
                                       const uint32_t *__restrict__ pts_slot,
                                       const float *__restrict__ pts_dist,
                                       float *__restrict__ dist_table,
                                       bool *__restrict__ mask) {
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= N)
        return;
    uint32_t batch_idx = blockIdx.y;
    uint32_t pt_idx_global = N * batch_idx + pt_idx;

    auto *min_dist_in_table = dist_table + T * batch_idx + pts_slot[pt_idx_global];
    if (*min_dist_in_table == pts_dist[pt_idx_global]) {  // this point with minimum distance to its voxel center.
        // remove the min_distance recorded in dist_table[pt_slot]
        // to guarantee only one point is masked in one voxel.
        // this can be carefully deleted as this rarely happens with nature data.
        float prev = atomicExch(min_dist_in_table, FLT_MAX);
        if (prev != FLT_MAX) {  // the first point with min distance in this voxel
            mask[pt_idx_global] = true;
        }
    }
}

__global__
void mask_out_to_output_and_table_batch(const uint32_t N, const uint32_t M, const uint32_t T,
                                        const bool *__restrict__ mask,
                                        const int64_t *__restrict__ mask_sum,
                                        const uint32_t *__restrict__ pts_slot,
                                        uint64_t *__restrict__ ind,
                                        uint64_t *__restrict__ ind_table) {
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= N) {
        return;
    }
    auto bt_idx = blockIdx.y;
    auto pt_idx_batch = N * bt_idx + pt_idx;
    auto output_ind = mask_sum[pt_idx_batch];
    if (mask[pt_idx_batch]) {
        if (output_ind < M) {
            ind[M * bt_idx + output_ind] = pt_idx;
        }
        // every active element in slot_table corresponding an index of output.
        ind_table[T * bt_idx + pts_slot[pt_idx_batch]] = output_ind;
    }
}

/***
 * @param sample_num  the number of points to be sampled in each batch
 * @param voxel initial voxel size
 * @param threshold the upper tolerance of sample_num
 * @param point_xyz point coords in float(B,N,3)
 * @param max_itr the allowed maximum iteration times
 * @return sampled_ind
 */
std::vector<at::Tensor> AdaptiveSamplingAndQuery(at::Tensor &point_xyz, //at::Tensor &idx_cnt, at::Tensor &idx,
                                                 int sample_num, float tolerance, int maximum_iterations,
                                                 float init_voxel_x, float init_voxel_y, float init_voxel_z) {
    uint32_t B = point_xyz.size(0), N = point_xyz.size(1), M = sample_num;
    uint32_t T = get_table_size(N, 2048);

    auto
            init_voxel = torch::tensor({{init_voxel_x, init_voxel_y, init_voxel_z}}, AT_TYPE(point_xyz, Float)),
            voxel_lower = torch::zeros({B, 3}, AT_TYPE(point_xyz, Float)),
            voxel_cur = init_voxel.repeat({B, 1}).contiguous(),
            voxel_upper = voxel_cur * 2.0,

            batch_mask = torch::zeros({B}, AT_TYPE(point_xyz, Bool)),
            hash_table = torch::empty({B, T}, AT_TYPE(point_xyz, Int)),
            dist_table = torch::full({B, T}, at::Scalar(FLT_MAX), AT_TYPE(point_xyz, Float)),
            ind_table = torch::empty({B, T}, AT_TYPE(point_xyz, Long)),
            point_slot = torch::empty({B, N}, AT_TYPE(point_xyz, Int)),
            point_dist = torch::empty({B, N}, AT_TYPE(point_xyz, Float)),
            sampled_mask = torch::zeros({B, N}, AT_TYPE(point_xyz, Bool)),
            sampled_num = torch::empty({B}, AT_TYPE(point_xyz, Int)),
            sampled_ind = torch::zeros({B, M}, AT_TYPE(point_xyz, Long));

    auto iter_num = 0;
    auto hash_empty = at::Scalar((int) 0xffffffff);
    auto blocks = BLOCKS2D(N, B), threads = THREADS();

    DECL_PTR(voxel_cur, float3, float);  // current voxel size.
    DECL_PTR(voxel_lower, float3, float);  // the lower bound of voxel size to be searched.
    DECL_PTR(voxel_upper, float3, float);  // the upper bound of voxel size to be searched.
    DECL_PTR(batch_mask, bool, bool);  // is this batch sampled completely?
    DECL_PTR(hash_table, uint32_t, int);  // the hash table for voxel coord inserting.
    DECL_PTR(dist_table, float, float);  // the table to store the minimum distance.
    DECL_PTR(point_xyz, float3, float);  // the coordination of each point.
    DECL_PTR(point_slot, uint32_t, int);  // the position in hash-table for each point.
    DECL_PTR(point_dist, float, float);  // the minimum distance from each point to its corresponding voxel center.
    DECL_PTR(sampled_ind, uint64_t, long); // the output sampled stacked indices.
    DECL_PTR(sampled_num, uint32_t, int);  // how many points have been sampled in each batch.
    // additional
    DECL_PTR(sampled_mask, bool, bool);  // mask of input points to indicate if it will be sampled.
    DECL_PTR(ind_table, uint64_t, long);  // the "input ind" field of hash table.


    // compute potential voxel size.
    do {
        sampled_num.zero_();
        hash_table.fill_(hash_empty);
        valid_voxel_batch<<< blocks, threads>>>(
                N, T,
                batch_mask_ptr,
                point_xyz_ptr,
                voxel_cur_ptr,
                hash_table_ptr,
                sampled_num_ptr
        );
        voxel_update_kernel<<<1, B>>>(
                B, M, tolerance,
                sampled_num_ptr,
                batch_mask_ptr,
                voxel_cur_ptr,
                voxel_lower_ptr,
                voxel_upper_ptr
        );
    } while (++iter_num <= maximum_iterations and not batch_mask.all().item().toBool());

    find_mini_dist_for_valid_voxels_batch<<< blocks, threads>>>(
            N, T,
            point_xyz_ptr,
            voxel_cur_ptr,
            hash_table_ptr,
            dist_table_ptr,
            point_slot_ptr,
            point_dist_ptr
    );

    mask_input_if_with_min_dist_batch<<< blocks, threads>>>(
            N, T,
            point_slot_ptr,
            point_dist_ptr,
            dist_table_ptr,
            sampled_mask_ptr
    );

    auto inclusive_prefix_sum = torch::cumsum(sampled_mask, -1);
    inclusive_prefix_sum -= 1;
    DECL_PTR(inclusive_prefix_sum, int64_t, int64_t);

    mask_out_to_output_and_table_batch<<<blocks, threads>>>(
            N, M, T,
            sampled_mask_ptr,
            inclusive_prefix_sum_ptr,
            point_slot_ptr,
            sampled_ind_ptr,
            ind_table_ptr);
//    assert(((sampled_num - 1) == inclusive_prefix_sum.index({"...", -1})).sum().item().toInt() == B);
    return {sampled_ind, voxel_cur, hash_table, ind_table};
}

__global__
void grid_query_batch(const uint32_t N, const uint32_t M, const uint32_t S, const uint32_t T, const float search_radius,
                      const float3 *__restrict__ queries,
                      const float3 *__restrict__ sources,
                      const float3 *__restrict__ voxels,
                      const uint32_t *__restrict__ hash_table,
                      const uint64_t *__restrict__ slots2queries,
                      uint64_t *__restrict__ indices,
                      uint32_t *__restrict__ nums) {

    uint32_t batch_idx = blockIdx.y;
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= N)
        return;

    auto queries_batch = queries + M * batch_idx;
    auto hash_table_batch = hash_table + T * batch_idx;
    auto slots2queries_batch = slots2queries + T * batch_idx;

    auto nums_batch = nums + M * batch_idx;  // int[B,M]
    auto indices_batch = indices + M * S * batch_idx;  // long[B,M,S]

    auto source = sources[N * batch_idx + pt_idx], v = voxels[batch_idx];
    int s_grid_x = roundf(source.x / v.x);
    int s_grid_y = roundf(source.y / v.y);
    int s_grid_z = roundf(source.z / v.z);

    int step_x = ceilf(search_radius / v.x + 0.5f) - 1;
    int step_y = ceilf(search_radius / v.y + 0.5f) - 1;
    int step_z = ceilf(search_radius / v.z + 0.5f) - 1;

    auto r2 = search_radius * search_radius;

    uint32_t MAX = T - 1;
    for (int q_grid_z = s_grid_z - step_z; q_grid_z <= s_grid_z + step_z; ++q_grid_z) {
        for (int q_grid_y = s_grid_y - step_y; q_grid_y <= s_grid_y + step_y; ++q_grid_y) {
            for (int q_grid_x = s_grid_x - step_x; q_grid_x <= s_grid_x + step_x; ++q_grid_x) {
                uint32_t key = coord_hash_32(q_grid_x, q_grid_y, q_grid_z);
                uint32_t slot = key & MAX;
                while (true) {
                    if (hash_table_batch[slot] == key) {  // hit a non-empty neighbour voxel.
                        auto query_idx = slots2queries_batch[slot];
                        if (query_idx < M) {  // but this voxel isn't sampled since we have sampled enough points.
                            auto offset = source - queries_batch[query_idx];
                            auto d2 = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
                            if (d2 <= r2 and nums_batch[query_idx] <= S) {
                                auto neighbor_idx_of_q = atomicAdd(nums_batch + query_idx, 1);
                                if (neighbor_idx_of_q < S) {
                                    indices_batch[query_idx * S + neighbor_idx_of_q] = pt_idx;
                                }
                            }
                        }
                        break;
                    }
                    if (hash_table_batch[slot] == kEmpty) {  // empty neighbour voxel.
                        break;
                    }
                    slot = (slot + 1) & MAX;
                }
            }
        }
    }
}

__global__
void pad_indices(const uint32_t M, const uint32_t S,
                 uint64_t *__restrict__ indices,
                 uint32_t *__restrict__ nums) {
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= M)
        return;

    uint32_t batch_idx = blockIdx.y;
    auto &num = nums[M * batch_idx + pt_idx];
    if (num > S) {
        num = S;
    }

    auto idx = indices + M * S * batch_idx + S * pt_idx;  // [B,M,S]
    if (num) {
        for (int l = 0; num < S; ++l, ++num) {
            idx[num] = idx[l];
        }
    }
}

/***
 * @note: 我们是对source并行，而不是对query并行，因此并行力度更大。并行后为O(K)而原来是O(N), 理想情况应该提速100-1000倍。
 * @param query_xyz coords of the points used to query.
 * @param source_xyz the points to be queries in.
 * @param hash_table hash_table[slot] indicates a voxel is non-empty and corresponding to query_xyz.
 * @param ind_table ind_table[slot] indicates the index of query_xyz in source_xyz.
 * @param voxels the number of grid steps the query_xyz used to search nearest points.
 * @param neighbours_num the number of nearest points the query_xyz need to query.
 * @param search_radius the radius of nearest points searching.
 */
std::vector<at::Tensor> QueryFromHash(at::Tensor &source_xyz,
                                      at::Tensor &query_xyz,
                                      at::Tensor &hash_table,
                                      at::Tensor &slots2queries,
                                      at::Tensor &voxels,
                                      const float search_radius,
                                      const int neighbours_num) {
    uint32_t B = source_xyz.size(0), S = neighbours_num;
    uint32_t N = source_xyz.size(1), M = query_xyz.size(1), T = hash_table.size(1);

    auto blocks = BLOCKS2D(N, B), threads = THREADS();  // for each source point.
    auto indices = torch::zeros({B, M, S}, AT_TYPE(source_xyz, Long));
    auto valid_nums = torch::zeros({B, M}, AT_TYPE(source_xyz, Int));

    DECL_PTR(query_xyz, float3, float);
    DECL_PTR(source_xyz, float3, float);
    DECL_PTR(hash_table, uint32_t, int);
    DECL_PTR(slots2queries, uint64_t, long);
    DECL_PTR(voxels, float3, float);
    DECL_PTR(indices, uint64_t, long);
    DECL_PTR(valid_nums, uint32_t, int);

    // query: th
    grid_query_batch<<<blocks, threads>>>(
            N, M, S, T, search_radius,
            query_xyz_ptr,
            source_xyz_ptr,
            voxels_ptr,
            hash_table_ptr,
            slots2queries_ptr,
            indices_ptr,
            valid_nums_ptr);

    // tail padding
    blocks = BLOCKS2D(M, B);
    pad_indices<<<blocks, threads>>>(M, S, indices_ptr, valid_nums_ptr);
    return {valid_nums, indices};
}
