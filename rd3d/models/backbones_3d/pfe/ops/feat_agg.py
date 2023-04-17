import torch
import torch.nn as nn

from .....utils.common_utils import ScopeTimer, apply1d
from .builder import aggregation
from typing import List


@aggregation.register_module('cat-mlps')
class CatAggregation(nn.Module):
    def __init__(self, aggregation_cfg, input_channels):
        super(CatAggregation, self).__init__()
        self.cfg = aggregation_cfg
        self.output_channels = sum(input_channels)
        if self.cfg.get('mlps', None) is not None:
            mlps_cfg = [self.output_channels] + self.cfg.mlps
            fc_list = [[nn.Linear(mlps_cfg[k], mlps_cfg[k + 1], bias=False),
                        nn.BatchNorm1d(mlps_cfg[k + 1]),
                        nn.ReLU()] for k in range(mlps_cfg.__len__() - 1)]
            self.mlps = nn.Sequential(*[ii for i in fc_list for ii in i])
            self.output_channels = self.cfg.mlps[-1]
        else:
            self.mlps = None

    def forward(self, feats_list: List[torch.tensor]) -> torch.tensor:
        new_features = torch.cat(feats_list, dim=-1)
        if self.mlps is not None:
            new_features = apply1d(self.mlps, new_features)
        return new_features


@aggregation.register_module('mlps-mask-sum')
class SumAggregation(nn.Module):
    def __init__(self, aggregation_cfg, input_channels: List[int]):
        super(SumAggregation, self).__init__()
        self.cfg = aggregation_cfg
        self.output_channels = self.cfg.mlps[-1]

        if self.cfg.get('mlps', None) is not None:
            self.mlps = nn.ModuleList()
            assert len(input_channels) > 0
            for channels in input_channels:
                mlps_cfg = [channels] + self.cfg.mlps
                fc_list = [[nn.Conv1d(mlps_cfg[k], mlps_cfg[k + 1], kernel_size=1, bias=False),
                            nn.BatchNorm1d(mlps_cfg[k + 1]),
                            nn.ReLU()] for k in range(mlps_cfg.__len__() - 1)]
                self.mlps.append(nn.Sequential(*[ii for i in fc_list for ii in i]))
        else:
            assert input_channels.count(input_channels[0]) == len(input_channels)
            self.mlps = None

    def forward(self, feats_list: List[torch.tensor], prob_list: List[torch.tensor], mask_list: List[torch.tensor]):
        """

        Args:
            feats_list: [ (C, active) ]
            prob_list: [ (B, M) ] differentialble soft mask
            mask_list: [ (B, M) ] hard mask
        Returns:
            new_feats: (B, C, M)
        """
        (b, m), c, k = prob_list[0].size(), self.output_channels, feats_list.__len__()
        new_feats = prob_list[0].new_zeros([k, c, b, m])

        # with TimeMeasurement(f"mlps-mask-sum-{[x.shape[0] for x in feats_list]}: ") as t:
        for g, (agg, feats, prob, mask) in enumerate(zip(self.mlps, feats_list, prob_list, mask_list)):
            if feats is not None and feats.shape[-1] > 1:
                new_feats[g, :, mask] = agg(feats.unsqueeze(0)).squeeze(0)
            if self.training:
                new_feats[g] *= prob.unsqueeze(0)
        new_feats = new_feats.permute(0, 2, 1, 3).sum(dim=0)

        return new_feats
