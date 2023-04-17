import torch
from .detector3d_template import Detector3DTemplate


class PointVote(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            self.assign_targets(batch_dict)
            loss, tb_dict, _ = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict
        else:
            if torch.onnx.is_in_onnx_export():
                return self.post_processing_export(batch_dict)
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

    def assign_targets(self, batch_dict):
        self.backbone_3d.assign_targets(batch_dict)
        self.point_head.assign_targets(batch_dict)

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}

        loss_backbone, tb_dict1 = self.backbone_3d.get_loss()
        loss_point, tb_dict2 = self.point_head.get_loss()
        loss = loss_point + loss_backbone

        tb_dict.update(tb_dict1)
        tb_dict.update(tb_dict2)
        return loss, tb_dict, disp_dict
