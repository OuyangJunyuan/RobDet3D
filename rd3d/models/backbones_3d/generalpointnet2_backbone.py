import torch
import torch.nn as nn
from easydict import EasyDict
from copy import deepcopy
from .pfe.point_set_abstraction import GeneralPointSetAbstraction
from .pfe.ops.builder import sampler


class GeneralPointNet2MSG(nn.Module):
    def __init__(self, model_cfg: EasyDict, input_channels: int, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.encoder_layers = nn.ModuleList()

        skip_channel_list = [input_channels - 3]

        for k, raw_sa_cfg in enumerate(self.model_cfg.get('ENCODER', [])):  # build encoder
            self.encoder_layers.append(GeneralPointSetAbstraction(deepcopy(raw_sa_cfg), skip_channel_list[k]))
            skip_channel_list.append(self.encoder_layers[-1].output_channels)

        self.num_point_features = skip_channel_list[-1]

        # build decoder
        if self.model_cfg.get('DECODER', None) is not None:
            self.decoder_layers = nn.ModuleList()
            raise NotImplementedError
        else:
            self.decoder_layers = None

        sampler_cfg = self.model_cfg.get('BATCHIFY_SAMPLE', None)
        self.sampler = sampler.from_cfg(sampler_cfg) if sampler_cfg is not None else None

    def split(self, points, bs=None):
        if points.dim() == 3:
            bs = points.shape[0]
            bid, xyz, feats = None, points[..., 0:3], points[..., 3:]
        elif points.dim() == 2:
            bs = points[-1, 0].int().item() + 1  # (n,[bid,x,y,z,...])
            if self.sampler is None:
                points = points.view(bs, -1, points.shape[-1])
                bid, xyz, feats = points[..., 0:1], points[..., 1:4], points[..., 4:]
            else:
                bid, xyz, feats = points[..., 0:1].long().contiguous(), points[..., 1:4].contiguous(), points[..., 4:]
                ind = self.sampler(xyz, bid, batch_size=bs)
                bid, xyz, feats = bid[ind].view(bs, -1, 1), xyz[ind].view(bs, -1, 3), feats[ind].view(bs, -1, 1)
        else:
            raise ValueError("the dimension of input points is wrong!")
        return bs, None if bid is None else bid.long().contiguous(), xyz.contiguous(), feats.contiguous()

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        """
        points = batch_dict['points']
        batch_size, bid, xyz, feats = self.split(points, batch_dict.get('batch_size', None))  # (B,N,1) (B,N,3) (B,N,C0)

        l_xyz, l_features = [xyz], [feats]
        for encoder in self.encoder_layers:
            li_xyz, li_features, bid = encoder(l_xyz[-1], l_features[-1], bid)
            l_xyz.append(li_xyz), l_features.append(li_features)

        if self.decoder_layers is None:
            output_layer = -1
        else:
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
            output_layer = 0

        batch_dict['batch_size'] = batch_size
        batch_dict['point_features'] = l_features[output_layer].contiguous()  # (B, C, N)
        batch_dict['point_coords'] = l_xyz[output_layer].contiguous()  # (B, N, 3)
        batch_dict['point_batch_id'] = bid
        return batch_dict

    #################################
    def assign_targets(self, batch_dict):
        for encoder in self.encoder_layers:
            encoder.assign_targets(batch_dict)

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        def get_submodule_loss(sm):
            if hasattr(sm, 'get_loss'):
                loss, tb_dict_ = sm.get_loss(tb_dict)
                tb_dict.update(tb_dict_)
                return loss
            return 0

        backbone_loss = 0
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    backbone_loss = backbone_loss + get_submodule_loss(m)
            backbone_loss = backbone_loss + get_submodule_loss(module)

        return backbone_loss, tb_dict
