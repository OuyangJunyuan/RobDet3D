from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

from . import pointnet2_batch_cuda as pointnet2


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative farthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply


@torch.no_grad()
def calc_dist_matrix_for_sampling(xyz: torch.Tensor, features: torch.Tensor = None,
                                  gamma: float = 1.0):
    dist = torch.cdist(xyz, xyz)

    if features is not None:
        dist += torch.cdist(features, features) * gamma

    return dist


@torch.no_grad()
def furthest_point_sample_matrix(matrix: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance with a pairwise distance matrix
    :param matrix: (B, N, N) tensor of dist matrix
    :param npoint: int, number of features in the sampled set
    :return:
         output: (B, npoint) tensor containing the set
    """
    assert matrix.is_contiguous()

    B, N, _ = matrix.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    pointnet2.furthest_point_sampling_matrix_wrapper(B, N, npoint, matrix, temp, output)
    return output


@torch.no_grad()
def furthest_point_sample_weights(xyz: torch.Tensor, weights: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum weighted distance
    Args:
        xyz: (B, N, 3), tensor of xyz coordinates
        weights: (B, N), tensor of point weights
        npoint: int, number of points in the sampled set
    Returns:
        output: (B, npoint) tensor containing the set
    """
    assert xyz.is_contiguous()
    assert weights.is_contiguous()

    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    pointnet2.furthest_point_sampling_weights_wrapper(B, N, npoint, xyz, weights, temp, output)
    return output


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
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

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


@torch.no_grad()
def ball_query_cnt(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
    """
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
    idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()

    pointnet2.ball_query_cnt_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx_cnt, idx)
    return idx_cnt, idx


@torch.no_grad()
def ball_query_dilated(radius_in: float, radius_out: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
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
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()
    idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

    pointnet2.ball_query_dilated_wrapper(B, N, npoint, radius_in, radius_out, nsample, new_xyz, xyz, idx_cnt, idx)
    return idx_cnt, idx


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class QueryAndGroupCnt(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx_cnt, idx = ball_query_cnt(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return idx_cnt, new_features


class QueryAndGroupDilated(nn.Module):
    def __init__(self, radius_in: float, radius_out: float, nsample: int, use_xyz: bool = True):
        """
        :param radius_in: float, radius of inner ball
        :param radius_out: float, radius of outer ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius_in, self.radius_out, self.nsample, self.use_xyz = radius_in, radius_out, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
            idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
        """
        idx_cnt, idx = ball_query_dilated(self.radius_in, self.radius_out, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B,3,npoint, nsample)=(B,3,N)op(B,npoint,nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)  # (B,npoint,3)->(B,3,npoint)->(B,3,npoint,1)->(B,3,np,ns)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return idx_cnt, new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class PointStructuringNet(nn.Module):
    def __init__(self,
                 npoint: int,
                 nsample: int,
                 mlps: List[int],
                 use_coords_augment: bool = True,
                 use_xyz: bool = True):
        """
        global features is no used here since there is not instance'points analysis.
        Args:
            npoint: the number of sampled points.
            nsample: the number of points to be grouped in local area.
            mlps: the setting of intermedia mlps channels.
            coords_augment:
        """
        super(PointStructuringNet, self).__init__()
        assert mlps.__len__() > 0

        self.npoint = npoint
        self.nsample = nsample
        self.mlps = self.make_fc_layers(mlps, input_channels=3, output_channels=npoint)
        self.use_coords_augment = use_coords_augment
        self.use_xyz = use_xyz

    def make_fc_layers(self, mlps_cfg: List[int], input_channels: int, output_channels: int):
        fc_layers = []
        for inner_output_channels in mlps_cfg:
            fc_layers.extend([
                nn.Conv1d(input_channels, inner_output_channels, bias=False, kernel_size=1),
                nn.BatchNorm1d(inner_output_channels),
                nn.ReLU(),
            ])
            input_channels = inner_output_channels
        fc_layers.append(nn.Conv1d(input_channels, output_channels, bias=True, kernel_size=1))
        return nn.Sequential(*fc_layers)

    def coords_augment(self, xyz: torch.Tensor):
        r = torch.norm(xyz[..., :3], dim=-1, keepdim=True)
        th = torch.acos(xyz[..., 2] / r).unsqueeze(-1)
        fi = torch.atan2(xyz[..., 1], xyz[..., 0]).unsqueeze(-1)
        return torch.cat((xyz, th, fi), -1)

    def gumbel_softmax(self, logits, dim=-1, temperature=0.001):
        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape)
            U = U.cuda()
            return -torch.log(-torch.log(U + eps) + eps)

        y = logits + sample_gumbel(logits.size())
        return torch.softmax(y / temperature, dim=dim)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None):
        """

        Args:
            xyz: [B,N,3]
            features: [B,C,N]

        Returns:
            new_xyz: [B,npoint,3]
            new_features: [B,3+C,npoints,nsample]

        """
        assert features is not None and self.use_xyz is True, "Cannot have not features and not use xyz as a feature!"

        xyz_aug = xyz if self.use_coords_augment else self.coords_augment(xyz[:, :3])  # [B,N,3]
        xyz_trans = xyz_aug.transpose(1, 2).contiguous()  # [B,3,N]

        # predict grouping probability
        sample_prob = torch.sigmoid(self.mlps(xyz_trans))  # [B,npoint,N]

        # group xyz by predicted probability
        group_indices = torch.topk(sample_prob, k=self.nsample, dim=-1)[1].int()  # [B,npoint,nsample]
        grouped_xyz = grouping_operation(xyz_trans, group_indices)  # [B,3,npoint,nsample]
        grouped_feat = None if features is None else grouping_operation(features, group_indices)  # [B,C,npoint,nsample]

        # # sample xyz and features depend on network state
        # if self.training:
        #     sample_prob = self.gumbel_softmax(sample_prob)  # [B,npoint,N]
        #     sampled_xyz = torch.matmul(sample_prob, xyz_aug)  # [B,npoint,3] = [B,npoint,N] * [B,N,3]
        #     sampled_xyz = sampled_xyz.transpose(1, 2)
        #     if grouped_feat is not None:
        #         sampled_feat = torch.matmul(sample_prob, features.transpose(1, 2)).transpose(1, 2)
        #         grouped_feat[..., 0] = sampled_feat
        # else:
        #     sampled_xyz = grouped_xyz[..., 0]
        #
        # grouped_xyz -= sampled_xyz.unsqueeze(-1)  # [B,3,npoint,nsample] -= [B,3,npoint]
        #
        # if self.use_xyz:
        #     new_features = grouped_xyz if features is None else torch.cat((grouped_xyz, grouped_feat), dim=1)
        # else:
        #     new_features = grouped_feat

        # return sampled_xyz, new_features


class PointSamplingNet(nn.Module):
    def __init__(self,
                 npoint: int,
                 mlps: List[List[int]],
                 use_coords_augment: bool = True):
        """
        global features is no used here since there is not instance'points analysis.
        Args:
            npoint: the number of sampled points.
            nsample: the number of points to be grouped in local area.
            mlps: the setting of intermedia mlps channels.
            coords_augment:
        """
        super(PointSamplingNet, self).__init__()
        assert mlps.__len__() > 0

        self.npoint = npoint
        self.use_coords_augment = use_coords_augment
        self.initial_dim = 6 if self.use_coords_augment else 3

        self.mlps = nn.ModuleList()
        input_channel = self.initial_dim
        for mlp in mlps[:-1]:
            mlp_layer = self.make_fc_layers(mlp, input_channels=input_channel)
            self.mlps.append(mlp_layer)
            input_channel = mlp[-1] + self.initial_dim  # point-wise + global + skip
        self.mlps.append(self.make_fc_layers(mlps[-1], input_channels=input_channel, output_channels=1))

        self.forward_dict = {}

    def make_fc_layers(self, mlps_cfg: List[int], input_channels: int, output_channels=None):
        fc_layers = []
        for inner_output_channels in mlps_cfg:
            fc_layers.extend([
                nn.Conv1d(input_channels, inner_output_channels, bias=False, kernel_size=1),
                nn.BatchNorm1d(inner_output_channels),
                nn.ReLU(),
            ])
            input_channels = inner_output_channels
        if output_channels is not None:
            fc_layers.append(nn.Conv1d(input_channels, output_channels, bias=True, kernel_size=1))
        return nn.Sequential(*fc_layers)

    def coords_augment(self, xyz: torch.Tensor):
        r = torch.norm(xyz[..., :3], p=2, dim=-1, keepdim=True)
        th = torch.acos(xyz[..., 2, None] / r)
        fi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])
        return torch.cat((xyz, r, th, fi), -1)

    def sample_gumbel(self, x, eps=1e-20):
        U = torch.rand_like(x)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax(self, logits, dim=-1, temperature=0.001):
        y = logits + self.sample_gumbel(logits.size())
        return torch.softmax(y / temperature, dim=dim)

    def gumbel_sigmoid(self, x):
        g1, g2 = self.sample_gumbel(x), self.sample_gumbel(x)
        t1, t2 = torch.exp(x + g1), torch.exp(g2)
        sigmoid = t1 / (t1 + t2)
        return sigmoid
        # return F.gumbel_softmax()

    def get_loss(self):

        xyz = self.forward_dict['xyz']
        psnet_xyz = self.forward_dict['sampled_xyz']
        # xyz_norm = xyz.norm(dim=-1, keepdim=True)
        # sampled_prob = xyz_norm.max() / (xyz + 1e-8)
        # prob = self.gumbel_sigmoid(torch.logit(sampled_prob))
        # selected = prob.topk(k=self.npoint, dim=1)[1]
        # psnet_xyz = xyz.gather(1, selected)

        fps_sampled_indices = farthest_point_sample(xyz, self.npoint)
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        fps_xyz = gather_operation(xyz_flipped, fps_sampled_indices).transpose(1, 2).contiguous()

        rand_xyz = xyz.gather(1, torch.randint(0, self.npoint, size=[*xyz.shape[:2], 1], device=xyz.device).repeat(1, 1,
                                                                                                                   3))

        def fps_dist_loss():
            return torch.min(torch.cdist(fps_xyz, psnet_xyz), dim=-1)[0].sum() / self.npoint

        def fps_label_loss():
            pred = self.forward_dict['pred_prob'].view(32, -1)
            label = torch.zeros_like(pred).scatter(-1, fps_sampled_indices.long(), 1.0)
            return torch.nn.functional.binary_cross_entropy(pred, label)

        def punish_closest():
            dist = torch.cdist(psnet_xyz, psnet_xyz)
            dist[:, range(dist.size(1)), range(dist.size(1))] = 1e8
            min_dist_indices = torch.min(dist, dim=-1)[1]
            min_dist_pts = psnet_xyz.gather(1, min_dist_indices[..., None].repeat(1, 1, 3)).detach()
            min_dist = (psnet_xyz - min_dist_pts).norm(dim=-1)
            b_id, p_id = torch.where(min_dist == 0)

            score = (1 / min_dist) + 1e-8
            score[b_id, p_id] = 0

            return score.sum() / self.npoint / dist.shape[0]

        def viz():
            import open3d
            import numpy as np

            vis = open3d.visualization.Visualizer()
            vis.create_window()

            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.ones(3)

            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd)

            def add_points(x, c):
                pts = open3d.geometry.PointCloud()
                pts.points = open3d.utility.Vector3dVector(x[:, :3])

                if c is None:
                    pts.colors = open3d.utility.Vector3dVector(np.ones((x.shape[0], 3)))
                else:
                    pts.colors = open3d.utility.Vector3dVector(np.ones_like(x) * np.array(c).reshape(1, 3))

                vis.add_geometry(pts)

            add_points(xyz[0, ...].cpu().detach().numpy() + np.array([[0.0, 50.0, 0.0]]), c=[0.2, 0.2, 0.2])
            add_points(psnet_xyz[0, ...].cpu().detach().numpy(), c=[1, 0, 0])
            add_points(fps_xyz[0, ...].cpu().detach().numpy() + np.array([[0.0, -50.0, 0.0]]), c=[0, 1, 0])
            add_points(rand_xyz[0, ...].cpu().detach().numpy() + np.array([[0.0, -100.0, 0.0]]), c=[0, 1, 0])
            vis.run()
            vis.destroy_window()

        viz() if getattr(self, 'is_viz', None) is not None else ()
        loss = fps_label_loss()
        return loss

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None):
        """

        Args:
            xyz: [B,N,3]
            features: [B,C,N]

        Returns:
            new_xyz: [B,npoint,3]
            new_features: [B,3+C,npoints,nsample]

        """

        xyz_aug = self.coords_augment(xyz[..., :3]) if self.use_coords_augment else xyz  # [B,N,3]
        xyz_trans = xyz_aug.transpose(1, 2).contiguous()  # [B,3,N]
        point_feats = xyz_trans
        for i in range(len(self.mlps) - 1):
            point_feats = self.mlps[i](point_feats)  # [B,C,N]
            global_feat = torch.max(point_feats, dim=-1, keepdim=True)[0]
            point_feats = torch.cat((point_feats + global_feat, xyz_trans), dim=1)
        point_feats = self.mlps[-1](point_feats)  # [B,C,N]

        # sample_probs = point_feats.sigmoid()
        sample_probs = self.gumbel_sigmoid(point_feats)
        sample_indices = sample_probs.topk(k=self.npoint)[1]

        hard_mask = torch.zeros_like(sample_probs).scatter(-1, sample_indices, 1.0)
        sampled_mask = (hard_mask - sample_probs).detach() + sample_probs
        sampled_mask_trans = sampled_mask.transpose(1, 2)

        masked_xyz = sampled_mask_trans * xyz
        sampled_xyz = masked_xyz.view(-1, 3)[sampled_mask_trans.view(-1).bool(), :].view((xyz.shape[0], -1, 3))
        self.forward_dict.update({'mask': sampled_mask_trans, 'xyz': xyz, 'sampled_xyz': sampled_xyz,
                                  'pred_prob': sample_probs})
        return sampled_xyz


try:
    import torch_scatter
    import numpy


    def sample_closest_to_voxel_center(points: torch.tensor, batch_size: int, shift: float, voxel_size: float = 1.0):
        """

        Args:
            points: [n,4](batch_idx,x,y,z]
            batch_size: int
            shift: coord shift
            voxel_size: float

        Returns:
            voxel_to_near_point_idx: sampled points' indices
            sampled_n_in_each_batch: how many points was sampled in each batch

        """
        voxel_size = points.new_tensor([voxel_size] * 3)
        batch_id, point_coord = points[..., :1].long(), points[..., 1:4]

        point_voxel_coord = ((point_coord + shift) / voxel_size).round().long()
        point_voxel_batch_coord = (batch_id << 54) + point_voxel_coord

        _, point_to_voxel_index = point_voxel_batch_coord.unique(dim=0, return_inverse=True)

        point_to_voxel_dist = (point_coord - point_voxel_coord * voxel_size).norm(dim=-1)
        _, voxel_to_near_point_idx = torch_scatter.scatter_min(point_to_voxel_dist, point_to_voxel_index, dim=0)

        voxel_to_near_point_idx_batch = batch_id[voxel_to_near_point_idx]
        sampled_n_in_each_batch = torch.eq(voxel_to_near_point_idx_batch, points.new_tensor(range(batch_size))) \
            .sum(dim=0)
        return voxel_to_near_point_idx, sampled_n_in_each_batch


    def sample_closest_to_voxel_center_masked(points: torch.tensor, mask: torch.tensor,
                                              batch_size: int, shift: float, voxel_size: float = 1.0):
        """

        Args:
            points: [n,4](batch_idx,x,y,z]
            mask: [n,] bool
            batch_size: int
            shift: float
            voxel_size: float

        Returns:

        """
        points_ = points[mask]
        raw_indices = torch.where(mask)[0]
        sampled_indices_, sampled_n_each_batch = sample_closest_to_voxel_center(points_, batch_size, shift, voxel_size)
        sampled_indices = raw_indices[sampled_indices_]
        return sampled_indices, sampled_n_each_batch


    def hierarchical_voxel_center_sampling(input_points: torch.tensor, npoint: int, voxel_size: float = 0.4):
        batch_size, num_points = input_points.shape[:2]
        batch_id = input_points.new_tensor(range(0, batch_size)).view(-1, 1).repeat(1, num_points)
        points = torch.hstack((batch_id.view(-1, 1), input_points.view(-1, 3)))

        sample_n_in_each_batch = points.new_tensor(batch_size * [npoint]).long()
        sample_indices_list = [[] for i in range(batch_size)]
        sample_mask = points.new_ones(points.shape[0], dtype=torch.bool)
        while True:
            # print_tensor("need_to_sample: ", sample_n_in_each_batch)
            # print(f"voxel_size: {voxel_size}")
            index, sampled_n = sample_closest_to_voxel_center_masked(points, sample_mask, batch_size,
                                                                     voxel_size=voxel_size)

            need_sample = torch.greater(sample_n_in_each_batch, 0)
            over_sample = torch.greater(sampled_n, sample_n_in_each_batch)

            sample_mask[index] = False
            index_list = index.split(tuple(sampled_n))
            for cur_bt in range(batch_size):
                if need_sample[cur_bt]:
                    if over_sample[cur_bt]:
                        need_sample_num = sample_n_in_each_batch[cur_bt]
                        # sample_indices_list[cur_bt].append(index_list[cur_bt][-need_sample_num:] - cur_bt * num_points)
                        sample_indices_list[cur_bt].append(index_list[cur_bt][:need_sample_num] - cur_bt * num_points)

                        sample_n_in_each_batch[cur_bt] = 0
                        sample_mask[cur_bt * num_points:(cur_bt + 1) * num_points] = False
                    else:
                        sample_indices_list[cur_bt].append(index_list[cur_bt] - cur_bt * num_points)
                        sample_n_in_each_batch[cur_bt] -= len(index_list[cur_bt])

            need_sample = torch.greater(sample_n_in_each_batch, 0)
            if need_sample.sum() > 0:
                voxel_size /= (numpy.pi + numpy.e) / 4
            else:
                return torch.vstack([torch.cat(sample_index_a_batch) for sample_index_a_batch in sample_indices_list])


    def hierarchical_voxel_sampling_batch(input_points: torch.tensor, npoint: int = 4096, voxel_size=2.5):
        batch_size, num_points = input_points.shape[:2]

        batch_id = input_points.new_tensor(range(batch_size)).repeat_interleave(num_points)
        points = torch.hstack((batch_id.unsqueeze(-1), input_points.view(-1, 3)))

        npoint_batch = points.new_tensor(batch_size * [npoint]).long()
        sample_mask = points.new_ones(points.shape[0], dtype=torch.bool)
        sample_mask_view = sample_mask.view(batch_size, -1)
        shift = 0

        sample_indices_list = [[] for _ in range(batch_size)]
        while torch.sum(npoint_batch > 0):  # some scenes in batch have not been completed yet.
            ind, num = sample_closest_to_voxel_center_masked(points, sample_mask, batch_size, shift,
                                                             voxel_size=voxel_size)
            sample_mask[ind] = False  # mark the sampled position
            ind_batch = ind.split([*num])

            for cur_bt in torch.where(npoint_batch > 0)[0]:
                if num[cur_bt] > npoint_batch[cur_bt]:
                    ratio = num[cur_bt] // npoint_batch[cur_bt]
                    ind_cur_bt = ind_batch[cur_bt][:npoint_batch[cur_bt] * ratio:ratio]
                    sample_mask_view[cur_bt] = False
                else:
                    ind_cur_bt = ind_batch[cur_bt]

                ind_cur_bt -= cur_bt * num_points
                npoint_batch[cur_bt] -= ind_cur_bt.shape[0]
                sample_indices_list[cur_bt].append(ind_cur_bt)

            voxel_size /= 2
            shift += voxel_size / 2

        return torch.vstack([torch.cat(sample_index_a_batch) for sample_index_a_batch in sample_indices_list])
except:
    pass
