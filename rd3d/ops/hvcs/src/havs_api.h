#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

at::Tensor potential_voxel_size_wrapper(
        const at::Tensor &point_xyz,
        const int &sample_num,
        const std::vector<float> &voxel,
        const float &threshold,
        const int &max_itr
);

at::Tensor potential_voxel_size_stack_wrapper(
        const at::Tensor &point_bid,
        const at::Tensor &point_xyz,
        const int &batch_size,
        const int &sample_num,
        const std::vector<float> &voxel,
        const float &threshold,
        const int &max_itr
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
voxel_size_experiments_wrapper(
        const at::Tensor &point_xyz,
        const int &sample_num,
        const std::vector<float> &voxel,
        const float &threshold,
        const int &max_itr
);

std::vector<at::Tensor> AdaptiveSamplingAndQuery(at::Tensor &point_xyz,
                                                 int sample_num,
                                                 float tolerance, int maximum_iterations,
                                                 float init_voxel_x,
                                                 float init_voxel_y,
                                                 float init_voxel_z);

std::vector<at::Tensor> QueryFromHash(at::Tensor &source_xyz,
                                      at::Tensor &query_xyz,
                                      at::Tensor &hash_table,
                                      at::Tensor &slots2queries,
                                      at::Tensor &voxels,
                                      const float search_radius,
                                      const int neighbours_num);