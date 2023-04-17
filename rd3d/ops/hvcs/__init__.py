from . import hvcs_cuda
from typing import List
import torch


class HAVSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                xyz: torch.Tensor,
                sample_num: int,
                voxel_size: List[float],
                tolerance: float,
                max_iter: int) -> torch.Tensor:
        indices = hvcs_cuda.potential_voxel_size(xyz, sample_num, voxel_size, tolerance, max_iter)
        # if torch.onnx.is_in_onnx_export():
        #     indices = indices.int()
        return indices

    @staticmethod
    def symbolic(g, xyz: torch.Tensor,
                 sample_num: int,
                 voxel_size: List[float],
                 tolerance: float,
                 max_iter: int):
        return g.op(
            "rd3d::HAVSampling", xyz,
            sample_num_i=sample_num,
            voxel_size_f=voxel_size,
            tolerance_f=tolerance,
            max_iter_i=max_iter
            # g.op("Constant", value_t=torch.tensor([sample_num], dtype=torch.int32)),
            # g.op("Constant", value_t=torch.tensor(voxel_size, dtype=torch.float32)),
            # g.op("Constant", value_t=torch.tensor([tolerance], dtype=torch.float32)),
            # g.op("Constant", value_t=torch.tensor([max_iter], dtype=torch.int32))
        )


class HAVSamplerForGridQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                xyz: torch.Tensor,
                sample_num: int,
                voxel_size: List[float],
                tolerance: float,
                max_iter: int) -> torch.Tensor:
        indices, voxel, hash_table, ind_table = \
            hvcs_cuda.adaptive_sampling_and_query(xyz, sample_num, tolerance, max_iter, *voxel_size)
        return indices, voxel, hash_table, ind_table

    @staticmethod
    def symbolic(g, xyz: torch.Tensor,
                 sample_num: int,
                 voxel_size: List[float],
                 tolerance: float,
                 max_iter: int):
        return g.op(
            "rd3d::HAVSamplingQ", xyz,
            sample_num_i=sample_num,
            voxel_size_f=voxel_size,
            tolerance_f=tolerance,
            max_iter_i=max_iter,
            outputs=4,
        )


hav_sampling = HAVSampler.apply
hav_sampling_for_query = HAVSamplerForGridQuery.apply
