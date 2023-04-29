import torch
from typing import List
from . import havs_cuda


class HAVSampler(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, xyz: torch.Tensor,
                sample_num: int, voxel_size: List[float],
                tolerance: float, max_iter: int,
                return_voxel_infos: bool = False, return_query_infos: bool = True):
        assert xyz.is_contiguous()

        indices = xyz.new_empty([xyz.shape[0], sample_num], dtype=torch.int64)
        infos = havs_cuda.havs_batch(
            xyz, indices, *voxel_size, tolerance, max_iter, return_voxel_infos, return_query_infos
        )
        if return_voxel_infos or return_query_infos:
            return indices, infos[0], infos[1], infos[2]
        else:
            return indices

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 xyz: torch.Tensor,
                 sample_num: int, voxel_size: List[float],
                 tolerance: float, max_iter: int,
                 return_voxel_infos: bool = False, return_query_infos: bool = False):

        return g.op(
            "rd3d::HAVSampling", xyz,
            num_sample_i=sample_num,
            init_voxel_f=voxel_size,
            tolerance_f=tolerance,
            max_iteration_i=max_iter
        ) if not return_query_infos else \
            g.op(
                "rd3d::HAVSamplingReturnHash", xyz,
                num_sample_i=sample_num,
                init_voxel_f=voxel_size,
                tolerance_f=tolerance,
                max_iteration_i=max_iter,
                outputs=4
            )


class GridQuery(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx,
                radius: float, nsample: int,
                xyz: torch.Tensor, new_xyz: torch.Tensor,
                voxels: torch.Tensor, voxel_hashes: torch.Tensor, hash2query: torch.Tensor):
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        queried_ids = xyz.new_empty([new_xyz.size(0), new_xyz.size(1), nsample], dtype=torch.int64)
        num_queried = xyz.new_empty([new_xyz.size(0), new_xyz.size(1)], dtype=torch.int32)

        havs_cuda.query_batch(
            new_xyz, xyz, queried_ids, num_queried,
            voxels, voxel_hashes, hash2query,
            radius, nsample
        )
        return num_queried, queried_ids

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 radius: float, nsample: int,
                 xyz: torch.Tensor, new_xyz: torch.Tensor,
                 voxels: torch.Tensor, voxel_hashes: torch.Tensor, hash2query: torch.Tensor):
        return g.op(
            "rd3d::GridBallQuery", xyz, new_xyz, voxels, voxel_hashes, hash2query,
            num_neighbor_i=nsample,
            radius_f=radius,
            outputs=2
        )


havs_batch = HAVSampler.apply
query_batch = GridQuery.apply
