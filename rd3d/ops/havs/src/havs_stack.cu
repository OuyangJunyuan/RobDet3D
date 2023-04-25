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


__global__
void InitVoxels(float3 init_voxel, float3 (*__restrict__ voxel_infos)[3]);

__global__
void InitHashTables(const uint32_t num_hash,
                    const uint32_t *__restrict__ batch_masks,
                    uint32_t *__restrict__ hash_tables) {
    if (batch_masks[blockIdx.y]) {
        return;
    }
    hash_tables[num_hash * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x] = kEmpty;
}

__global__
void CountNonEmptyVoxel(const uint32_t num_src, const uint32_t num_hash,
                        const uint32_t *__restrict__ batch_masks,
                        const float3 *__restrict__ sources, const float3 (*__restrict__ voxel_infos)[3],
                        uint32_t *__restrict__ hash_tables, uint32_t *__restrict__ num_sampled) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_masks[bid] || pid >= num_src) {
        return;
    }

    const auto voxel = voxel_infos[bid][1];
    const auto point = sources[num_src * bid + pid];
    const auto table = hash_tables + num_hash * bid;

    const uint32_t hash_key = coord_hash_32((int) roundf(point.x / voxel.x),
                                            (int) roundf(point.y / voxel.y),
                                            (int) roundf(point.z / voxel.z));

    const uint32_t kHashMax = num_hash - 1;
    uint32_t hash_slot = hash_key & kHashMax;

    while (true) {
        const uint32_t old = atomicCAS(table + hash_slot, kEmpty, hash_key);
        if (old == hash_key) {
            return;
        }
        if (old == kEmpty) {
            atomicAdd(num_sampled + bid, 1);
            return;
        }
        hash_slot = (hash_slot + 1) & kHashMax;
    }
}

__global__
void UpdateVoxelsSizeIfNotConverge(const uint32_t num_batch, const uint32_t num_trg,
                                   const uint32_t lower_bound, const uint32_t upper_bound,
                                   uint32_t *__restrict__ batch_masks,
                                   float3 (*__restrict__ voxel_infos)[3],
                                   uint32_t *__restrict__ num_sampled) {
    uint32_t bid = threadIdx.x;
    if (batch_masks[bid])
        return;

    const auto num = num_sampled[bid];
    if (lower_bound <= num and num <= upper_bound) {   // fall into tolerance.
        batch_masks[bid] = 1;
        atomicAdd(&batch_masks[num_batch], 1);
        dbg("%d", num);
        dbg("%f %f %f", voxel_infos[bid].c.x, voxel_infos[bid].c.y, voxel_infos[bid].c.z);
    } else {  // has not converged yet.
        if (num > num_trg) {
            voxel_infos[bid][0] = voxel_infos[bid][1];
        }
        if (num < num_trg) {
            voxel_infos[bid][2] = voxel_infos[bid][1];
        }
        // update current voxel by the average of left and right voxels.
        voxel_infos[bid][1] = (voxel_infos[bid][0] + voxel_infos[bid][2]) / 2.0f;
        num_sampled[bid] = 0;
    }
}

__global__
void FindMiniDistToCenterForEachNonEmptyVoxels(const uint32_t num_src, const uint32_t num_hash,
                                               const float3 *__restrict__ sources,
                                               const float3 (*__restrict__ voxel_infos)[3],
                                               uint32_t *__restrict__ hash_tables,
                                               float *__restrict__ dist_tables,
                                               uint32_t *__restrict__ point_slots,
                                               float *__restrict__ point_dists) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const auto pid_global = num_src * bid + pid;
    const auto point = sources[pid_global];
    const auto voxel = voxel_infos[bid][1];

    const auto coord_x = roundf(point.x / voxel.x);
    const auto coord_y = roundf(point.y / voxel.y);
    const auto coord_z = roundf(point.z / voxel.z);
    const auto d1 = point.x - coord_x * voxel.x;
    const auto d2 = point.y - coord_y * voxel.y;
    const auto d3 = point.z - coord_z * voxel.z;
    const auto noise = (float) pid * FLT_MIN;  // to ensure all point distances are different.
    const auto dist = d1 * d1 + d2 * d2 + d3 * d3 + noise;
    point_dists[pid_global] = dist;

    const auto dist_table = dist_tables + num_hash * bid;
    const auto hash_table = hash_tables + num_hash * bid;
    const uint32_t hash_key = coord_hash_32((int) coord_x, (int) coord_y, (int) coord_z);

    const uint32_t kHashMax = num_hash - 1;
    uint32_t hash_slot = hash_key & kHashMax;
    while (true) {
        const uint32_t old = atomicCAS(hash_table + hash_slot, kEmpty, hash_key);
        assert(old != kEmpty); // should never meet.
        if (old == hash_key) {
            atomicMin(dist_table + hash_slot, dist);
            point_slots[pid_global] = hash_slot;
            return;
        }
        hash_slot = (hash_slot + 1) & kHashMax;
    }
}

__global__
void MaskSourceWithMinimumDistanceToCenter(const uint32_t num_src, const uint32_t num_hash,
                                           const uint32_t *__restrict__ point_slots,
                                           const float *__restrict__ point_dists,
                                           const float *__restrict__ dist_tables,
                                           uint8_t *__restrict__ point_masks) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const uint32_t pid_global = num_src * bid + pid;
    const auto min_dist = dist_tables[num_hash * bid + point_slots[pid_global]];
    point_masks[pid_global] = min_dist == point_dists[pid_global];
}

inline
void ExclusivePrefixSum(const uint32_t num_batch, const uint32_t num_src, const uint32_t num_hash,
                        void *temp_mem, uint8_t *point_masks, uint32_t *point_masks_sum, cudaStream_t stream) {
    size_t temp_mem_size = num_batch * num_hash;  // must be higher than expected.

    for (int bid = 0; bid < num_batch; ++bid) {
        cub::DeviceScan::ExclusiveSum(
                temp_mem, temp_mem_size,
                point_masks + bid * num_src,
                point_masks_sum + bid * num_src,
                num_src, stream
        );
    }
}

__global__
void MaskOutSubsetIndices(const uint32_t num_src, const uint32_t num_trg,
                          const uint8_t *__restrict__ point_masks,
                          const uint32_t *__restrict__ point_masks_sum,
                          uint64_t *__restrict__ sampled_ids) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const auto pid_global = num_src * bid + pid;
    const auto mask_sum = point_masks_sum[pid_global];
    if (point_masks[pid_global] and mask_sum < num_trg) {
        sampled_ids[num_trg * bid + mask_sum] = pid;
    }
}

__global__
void MaskOutSubsetIndices(const uint32_t num_src, const uint32_t num_trg, const uint32_t num_hash,
                          const uint32_t *__restrict__ point_slots,
                          const uint8_t *__restrict__ point_masks,
                          const uint32_t *__restrict__ point_masks_sum,
                          uint64_t *__restrict__ sampled_ids, uint32_t *__restrict__ hash2subset) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;

    if (pid >= num_src) {
        return;
    }
    const auto pid_global = num_src * bid + pid;
    const auto mask_sum = point_masks_sum[pid_global];
    if (point_masks[pid_global] and mask_sum < num_trg) {
        sampled_ids[num_trg * bid + mask_sum] = pid;
        hash2subset[num_hash * bid + point_slots[pid_global]] = mask_sum;
    }
}

void HAVSamplingBatchLauncher(const int num_batch, const int num_src,
                              const int num_trg, const int num_hash,
                              const float3 init_voxel, const float tolerance, const int max_iterations,
                              const float3 *sources, uint32_t *batch_masks,
                              uint32_t *num_sampled, float3 (*voxel_infos)[3],
                              uint32_t *hash_tables, float *dist_tables,
                              uint32_t *point_slots, float *point_dists, uint8_t *point_masks,
                              uint64_t *sampled_ids,
                              const bool return_hash2subset = false,
                              cudaStream_t stream = nullptr) {
    InitVoxels<<<1, num_batch, 0, stream>>>(init_voxel, voxel_infos);

    const auto src_grid = BLOCKS1D(num_src);
    const auto table_grid = BLOCKS1D(num_hash);
    const auto block = THREADS();

    const auto lower_bound = uint32_t((float) num_trg * (1.0f + 0.0f));
    const auto upper_bound = uint32_t((float) num_trg * (1.0f + tolerance));

    uint32_t num_complete = 0;
    uint32_t cur_iteration = 1;
    while (max_iterations >= cur_iteration++ and num_complete != num_batch) {
        InitHashTables<<<table_grid, block, 0, stream>>>(
                num_hash, batch_masks, hash_tables
        );
        CountNonEmptyVoxel<<<src_grid, block, 0, stream>>>(
                num_src, num_hash, batch_masks, sources, voxel_infos, hash_tables, num_sampled
        );

        if (max_iterations >= cur_iteration) {  // voxels should not be updated in last iteration.
            UpdateVoxelsSizeIfNotConverge<<<1, num_batch, 0, stream>>>(
                    num_batch, num_trg, lower_bound, upper_bound, batch_masks, voxel_infos, num_sampled
            );
        }
        cudaMemcpyAsync(
                &num_complete, &batch_masks[num_batch],
                sizeof(num_complete), cudaMemcpyDeviceToHost, stream
        );
        cudaStreamSynchronize(stream);
    }
    FindMiniDistToCenterForEachNonEmptyVoxels<<< src_grid, block, 0, stream>>>(
            num_src, num_hash, sources, voxel_infos, hash_tables, dist_tables, point_slots, point_dists
    );
    MaskSourceWithMinimumDistanceToCenter<<<src_grid, block, 0, stream>>>(
            num_src, num_hash, point_slots, point_dists, dist_tables, point_masks
    );

    auto *temp_mem = (void *) dist_tables;  // reuse dist_tables as temporary memories.
    auto *point_masks_sum = (uint32_t *) point_dists;  // reuse point_dists as point_masks_sum
    ExclusivePrefixSum(
            num_batch, num_src, num_hash, temp_mem, point_masks, point_masks_sum, stream
    );
    if (return_hash2subset) {
        auto *hash2subset = (uint32_t *) dist_tables;  // reuse dist_tables as hash2subset.
        // cudaMemsetAsync(hash2subset, 0x00, num_batch * num_hash * sizeof(uint32_t), stream);
        MaskOutSubsetIndices<<<src_grid, block, 0, stream>>>(
                num_src, num_trg, num_hash, point_slots, point_masks, point_masks_sum, sampled_ids, hash2subset
        );
    } else {
        MaskOutSubsetIndices<<<src_grid, block, 0, stream>>>(
                num_src, num_trg, point_masks, point_masks_sum, sampled_ids
        );
    }
}


std::vector<at::Tensor> HAVSamplingStackWrapper(at::Tensor &sources,  // [in]
                                                at::Tensor &batch_ids,  // [in]
                                                at::Tensor &sampled_ids,  // [out]
                                                const long num_batch,
                                                const double init_voxel_x,
                                                const double init_voxel_y,
                                                const double init_voxel_z,
                                                const double tolerance,
                                                const long max_iterations,
                                                const bool return_sample_infos = false,
                                                const bool return_query_infos = false) {
    const int64_t num_src = sources.size(0);
    const int64_t num_trg = sampled_ids.size(0);
    const int64_t num_hash = get_table_size(num_src);
    const float3 init_voxel = {(float) init_voxel_x,
                               (float) init_voxel_y,
                               (float) init_voxel_z};

    auto batch_masks_tensor = sources.new_empty({num_batch + 1}, TorchType<int>);
    auto num_sampled_tensor = sources.new_empty({num_batch}, TorchType<int>);
    auto hash_tables_tensor = sources.new_empty({num_hash}, TorchType<int>);
    auto dist_tables_tensor = sources.new_empty({num_hash}, TorchType<float>);
    auto point_slots_tensor = sources.new_empty({num_src}, TorchType<int>);
    auto point_dists_tensor = sources.new_empty({num_src}, TorchType<float>);
    auto point_masks_tensor = sources.new_empty({num_src}, TorchType<char>);
    auto voxel_infos_tensor = sources.new_empty({num_batch, 3 * 3}, TorchType<float>);

    batch_masks_tensor.zero_();
    num_sampled_tensor.zero_();
    dist_tables_tensor.fill_(at::Scalar(FLT_MAX));

    HAVSamplingBatchLauncher(
            num_batch, num_src, num_trg, num_hash,
            init_voxel, tolerance, max_iterations,
            (float3 *) sources.data_ptr(),
            (uint32_t *) batch_masks_tensor.data_ptr(),
            (uint32_t *) num_sampled_tensor.data_ptr(),
            (float3 (*)[3]) voxel_infos_tensor.data_ptr(),
            (uint32_t *) hash_tables_tensor.data_ptr(),
            (float *) dist_tables_tensor.data_ptr(),
            (uint32_t *) point_slots_tensor.data_ptr(),
            (float *) point_dists_tensor.data_ptr(),
            (uint8_t *) point_masks_tensor.data_ptr(),
            (uint64_t *) sampled_ids.data_ptr(),
            return_query_infos
    );

    auto ret = std::vector<at::Tensor>{voxel_infos_tensor};
    if (return_sample_infos) {
        ret.emplace_back(num_sampled_tensor);  // the number of points actually sampled
        ret.emplace_back(point_dists_tensor);  // mask of which sources point is sampled.
    }
    if (return_query_infos) {
        ret.emplace_back(hash_tables_tensor);
        ret.emplace_back(dist_tables_tensor);
    }
    return ret;
}



