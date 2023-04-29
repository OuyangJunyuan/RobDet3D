import torch
from typing import List

from .builder import querier
from .....ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda as cuda_ops_batch
from .....ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda as cuda_ops_stack


@torch.no_grad()
@querier.register_module('ball_old')
def basic_ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, **kwargs):
    """
    :param ctx:
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

    cuda_ops_batch.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
    return idx.long()


@torch.no_grad()
@querier.register_module('ball')
def ball_query_cnt(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, **kwargs):
    class BallQuery(torch.autograd.Function):

        @staticmethod
        @torch.no_grad()
        def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
            assert new_xyz.is_contiguous()
            assert xyz.is_contiguous()

            B, N, _ = xyz.size()
            npoint = new_xyz.size(1)
            idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
            idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()

            cuda_ops_batch.ball_query_cnt_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx_cnt, idx)
            return idx_cnt, idx.long()

        @staticmethod
        def symbolic(g: torch._C.Graph, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
            return g.op(
                "rd3d::BallQuery", xyz, new_xyz,
                nsample_i=nsample,
                radius_f=radius,
                outputs=2
            )

    return BallQuery.apply(radius, nsample, xyz, new_xyz)


@torch.no_grad()
@querier.register_module('shell', 'ball_dilated')
def ball_query_dilated(radius: List[float], nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, **kwargs):
    """
    :param radius_in: float, radius of the inner balls
    :param radius_out: float, radius of the outer balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    radius_in, radius_out = radius
    assert radius_in < radius_out
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()
    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()
    idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

    cuda_ops_batch.ball_query_dilated_wrapper(B, N, npoint, radius_in, radius_out, nsample, new_xyz, xyz, idx_cnt,
                                              idx)
    return idx_cnt, idx.long()


@torch.no_grad()
@querier.register_module('ball_stack')
def ball_query_stack(radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                     new_xyz: torch.Tensor, new_xyz_batch_cnt, **kwargs):
    """
    Args:
        ctx:
        radius: float, radius of the balls
        nsample: int, maximum number of features in the balls
        xyz: (N1 + N2 ..., 3) xyz coordinates of the features
        xyz_batch_cnt: (batch_size), [N1, N2, ...]
        new_xyz: (M1 + M2 ..., 3) centers of the ball query
        new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

    Returns:
        idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert new_xyz_batch_cnt.is_contiguous()
    assert xyz.is_contiguous()
    assert xyz_batch_cnt.is_contiguous()

    B = xyz_batch_cnt.shape[0]
    M = new_xyz.shape[0]
    idx = torch.cuda.IntTensor(M, nsample).zero_()

    cuda_ops_stack.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
    empty_ball_mask = (idx[:, 0] == -1)
    idx[empty_ball_mask] = 0
    return empty_ball_mask, idx.long()


@torch.no_grad()
@querier.register_module('grid_ball')
def ball_query_cnt_by_grid(radius: float, nsample: int,
                           xyz: torch.Tensor, new_xyz: torch.Tensor,
                           voxel_sizes: torch.Tensor, hash_tables: torch.Tensor, subset_tables: torch.Tensor, **kwargs):
    from .....ops.havs import query_batch
    return query_batch(radius, nsample, xyz, new_xyz, voxel_sizes, hash_tables, subset_tables)
