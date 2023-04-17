
#include "common.h"

/**
 * @note shared version
 */
__global__
void voxel_update_kernel(const uint32_t batch_size, const uint32_t sample_num, const float threshold,
                         const uint32_t *__restrict__ sampled_num, bool *__restrict__ batch_mask,
                         float3 *__restrict__ voxel_cur, float3 *__restrict__ voxel_lower,
                         float3 *__restrict__ voxel_upper) {
    uint32_t batch_id = threadIdx.x;
    if (batch_id >= batch_size || batch_mask[batch_id]) return;

    uint32_t num = sampled_num[batch_id];
    float upper_bound = sample_num * (1.0f + threshold), lower_bound = sample_num * 1.0f;
    if (upper_bound >= num and num >= lower_bound) {
        batch_mask[batch_id] = true;
//        printf("%d %d %f %f %f\n", batch_id, num, voxel_cur[batch_id].x, voxel_cur[batch_id].y, voxel_cur[batch_id].z);
        return;
    }
//    printf("%d %d %f %f %f\n", batch_id, num, voxel_cur[batch_id].x, voxel_cur[batch_id].y, voxel_cur[batch_id].z);

    if (num > sample_num) voxel_lower[batch_id] = voxel_cur[batch_id];
    if (num < sample_num) voxel_upper[batch_id] = voxel_cur[batch_id];
    voxel_cur[batch_id] = (voxel_lower[batch_id] + voxel_upper[batch_id]) / 2;
}