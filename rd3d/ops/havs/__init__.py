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
        return indices, infos

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 xyz: torch.Tensor,
                 sample_num: int, voxel_size: List[float],
                 tolerance: float, max_iter: int,
                 return_voxel_infos: bool = False, return_query_infos: bool = False):
        return g.op(
            "rd3d::HAVSampling", xyz,
            sample_num_i=sample_num,
            voxel_size_f=voxel_size,
            tolerance_f=tolerance,
            max_iter_i=max_iter
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
            "rd3d::GridBallQuery", xyz, new_xyz, voxel_hashes, hash2query, voxels,
            nsample_i=nsample,
            radius_f=radius,
            outputs=2
        )


havs_batch = HAVSampler.apply
query_batch = GridQuery.apply