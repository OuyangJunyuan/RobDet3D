import pickle

import numba
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from .instance_bank import Bank
from .points_filling import PointCloudFilling
from ...api.dist import on_rank0, all_gather_object, barrier
from ...utils.base import Hook, replace_attr, when, merge_dicts


class SS3DHookHelper:
    def __init__(self):
        self.aug_scene_preds_dict = {}
        self.raw_scene_preds_dict = {}
        self.instance_bank = None
        self.bg_holes_boxes = {}
        self.global_augments = None
        self.insert_points_filling = None
        self.enable = False

        self.root_dir = None
        self.class_names = None

    @staticmethod
    def load_data_to_numpy(args):
        for arg in args:
            for k in arg:
                arg[k] = arg[k].detach().cpu().numpy()
        return args

    def new_aug_queue_with_pfa(self, run, augmentor):
        pfa = PointCloudFilling(**run.ss3d.points_filling_augment,
                                root_dir=self.root_dir, class_names=self.class_names, bank=self.instance_bank)
        new_queue = [pfa] + augmentor.data_augmentor_queue
        return new_queue

    def hack_train_predicts(self, model):
        if not hasattr(SS3DHook.hack_train_predicts, 'results'):
            model_head = list(model.children())[-1]
            if hasattr(model_head, 'train_dict'):
                results = model_head.train_dict
            elif hasattr(model_head, 'forward_ret_dict'):
                results = model_head.forward_ret_dict
            else:
                raise KeyError
            setattr(SS3DHook.hack_train_predicts, 'results', results)
        train_preds = SS3DHook.hack_train_predicts.results
        key_map = {'box_preds': 'box_preds', 'cls_preds': 'cls_preds'}
        if 'box_preds' not in train_preds: key_map['box_preds'] = 'rcnn_reg'
        if 'cls_preds' not in train_preds: key_map['cls_preds'] = 'rcnn_cls'
        return {ok: train_preds[ik].detach().clone() for ok, ik in key_map.items()}

        # cat_scores = np.vstack([preds1['pred_scores'][None, ind1], preds2['pred_scores'][None, ind2]])
        # better = argmax(cat_scores, 0)
        # higher_scores = cat_scores[better, range(ind1.size)]
        # mask = higher_scores > score_thr_high
        # ind1, ind2, better = ind1[mask], ind2[mask], better[mask]
        #
        # pairs = (pairs[0][mask], pairs[1][mask])
        # better_one = better_one[mask]
        # output = {}
        # for k in preds1.keys():
        #     value = np.vstack([preds1[k][None, pairs[0]], preds2[k][None, pairs[1]]])
        #     ind = better_one[None, :, None] if value.ndim == 3 else better_one[None, :]
        #     output[k] = np.take_along_axis(value, ind, axis=0)[0]
        # return output

    def reliable_background_mining(self, batch, model, mine_bg_cfg):
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
        bs = batch['batch_size']
        pred_dicts = self.hack_train_predicts(model)
        pred_scores = pred_dicts['cls_preds'].max(dim=-1)[0].sigmoid().view(bs, -1, 1)
        pred_boxes = pred_dicts['box_preds'].view(bs, pred_scores.shape[1], -1)

        infos = {}
        for boxes, scores, gts, fid, aug_log in zip(pred_boxes, pred_scores,
                                                    batch['gt_boxes'], batch['frame_id'], batch['aug_logs']):
            pseudo = boxes.new_tensor(np.array(merge_dicts(self.instance_bank.bk_infos[fid])['box3d_lidar']))

            score_mask = scores.squeeze() > mine_bg_cfg.score_threshold
            boxes = boxes[score_mask]
            collision_mask = boxes_iou_bev(boxes[..., :7], gts[..., :7]).any(dim=-1)
            boxes = boxes[torch.logical_not(collision_mask)]
            collision_mask = boxes_iou_bev(boxes[..., :7], pseudo[..., :7]).any(dim=-1)
            boxes = boxes[torch.logical_not(collision_mask)]

            self.global_augments.invert({'gt_boxes': boxes}, aug_log)
            infos[fid] = boxes.cpu().numpy()
            if mine_bg_cfg.get('visualize', False):
                from ...utils.viz_utils import viz_scenes
                ####################
                aug_points = batch['points'].view(-1, 16384, 5)[0].clone()
                aug_gt_boxes = batch['gt_boxes'][0].clone()
                ####################
                aug_pred_boxes = pred_boxes[0].clone()
                ####################
                inv_aug_points = self.global_augments.invert({'points': aug_points[:, 1:4].clone()},
                                                             batch['aug_logs'][0])['points']
                aug_unreliable_pred_boxes = infos[batch['frame_id'][0]].copy()
                ####################
                viz_scenes((aug_points, aug_gt_boxes),
                           (aug_points, aug_pred_boxes),
                           (inv_aug_points, aug_unreliable_pred_boxes),
                           offset=[0, 50, 0])
        return infos


class MineUnlabeledInstance:
    def __init__(self, cfg, ins_bank):
        from ...datasets.augmentor.transforms import AugmentorList

        self.cfg = cfg
        self.ins_bank = ins_bank
        self.aug_scene_preds_dict = {}
        self.raw_scene_preds_dict = {}
        self.global_augments = AugmentorList(self.cfg.global_augments)

    @staticmethod
    def load_data_to_numpy(args):
        for arg in args:
            for k in arg:
                arg[k] = arg[k].detach().cpu().numpy()
        return args

    def load(self, root):
        aug_path = root / 'ss3d/aug_scene_preds_dict.pkl'
        raw_path = root / 'ss3d/raw_scene_preds_dict.pkl'
        ret = aug_path.exists() and raw_path.exists()
        if ret:
            self.aug_scene_preds_dict = pickle.load(open(aug_path, 'rb'))
            self.raw_scene_preds_dict = pickle.load(open(raw_path, 'rb'))
        return ret

    @on_rank0
    def save(self, root):
        aug_path = root / 'ss3d/aug_scene_preds_dict.pkl'
        raw_path = root / 'ss3d/raw_scene_preds_dict.pkl'
        pickle.dump(self.aug_scene_preds_dict, open(aug_path, 'wb'))
        pickle.dump(self.raw_scene_preds_dict, open(raw_path, 'wb'))

    @torch.no_grad()
    def predict_aug_and_raw_scenes(self, dataloader, model):
        from tqdm import tqdm

        root_dir = dataloader.dataset.root_path

        if self.cfg.cache and self.load(root_dir):
            return

        model.eval()
        augmentor = dataloader.dataset.data_augmentor
        with replace_attr(augmentor, 'data_augmentor_queue', []):
            for batch_dict in tqdm(iterable=dataloader, desc='raw scenes', leave=False, dynamic_ncols=True):
                dataloader.dataset.load_data_to_gpu(batch_dict)
                pred_dict, info_dict = model(batch_dict)
                self.raw_scene_preds_dict.update({
                    fid: (pred, None) for fid, pred in
                    zip(batch_dict['frame_id'], self.load_data_to_numpy(pred_dict))
                })

        with replace_attr(augmentor, 'data_augmentor_queue', [self.global_augments]):
            for batch_dict in tqdm(iterable=dataloader, desc='aug scenes', leave=False, dynamic_ncols=True):
                dataloader.dataset.load_data_to_gpu(batch_dict)
                pred_dict, info_dict = model(batch_dict)
                self.aug_scene_preds_dict.update({
                    fid: (pred, logs) for fid, pred, logs in
                    zip(batch_dict['frame_id'], self.load_data_to_numpy(pred_dict),
                        batch_dict['augment_logs'])
                })

        all_gather_object(raw_scene_preds_dict=self.raw_scene_preds_dict,
                          aug_scene_preds_dict=self.aug_scene_preds_dict)
        self.save(root_dir)

    @staticmethod
    def score_based_filter(*args, score_thr=0.9, field='pred_scores'):
        for arg in args:
            keep = arg[field] > score_thr
            for key in arg.keys():
                arg[key] = arg[key][keep]

    @staticmethod
    def iou_guided_suppression(preds1, preds2, iou_thr=0.9):
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

        boxes1, boxes2 = preds1['pred_boxes'], preds2['pred_boxes']
        n, m = boxes1.shape[0], boxes2.shape[0]
        if n or m == 0:
            return
        iou = boxes_bev_iou_cpu(boxes1, boxes2)
        ind1, ind2 = np.arange(n), np.argmax(iou, -1)
        pairs_iou = iou[ind1, ind2]
        keep = pairs_iou > iou_thr

        for arg in [preds1, preds2]:
            for key in arg.keys():
                arg[key] = arg[key][keep]

    @on_rank0
    def missing_annotated_mining(self, dataloader):
        from ...utils.viz_utils import viz_scenes
        for fid in tqdm(iterable=self.aug_scene_preds_dict, desc='mining(filter)', leave=False, dynamic_ncols=True):
            raw_pred, *no_logs = self.raw_scene_preds_dict[fid]
            aug_pred, aug_logs = self.aug_scene_preds_dict[fid]

            raw_points = getattr(dataloader.dataset, self.cfg.get_points_func)(fid)

            self.global_augments.invert({'gt_boxes': aug_pred['pred_boxes']}, aug_logs)

            if self.cfg.get('visualize', False):
                print(raw_pred['pred_scores'])
                print(aug_pred['pred_scores'])
                pts_bank, boxes_bank, gt_masks_bank = self.ins_bank.get_scene(fid)
                self.global_augments.invert({'points': aug_pred['pred_boxes']}, aug_logs)
                viz_scenes((raw_points, raw_pred['pred_boxes']),
                           (raw_points, aug_pred['pred_boxes']),
                           (raw_points, boxes_bank), offset=[0, 50, 0])

            self.score_based_filter(raw_pred, aug_pred, score_thr=self.cfg.score_threshold_high)
            self.iou_guided_suppression(raw_pred, aug_pred, iou_thr=self.cfg.iou_threshold)

            reliable_pred = raw_pred

            if self.cfg.get('visualize', False):
                viz_scenes((raw_points, reliable_pred['pred_boxes']))

            if reliable_pred['pred_boxes'].shape[0] > 0:
                self.ins_bank.try_insert(fid, raw_points, **reliable_pred)

        self.ins_bank.save_to_disk()

    def mine_unlabeled_instance_one_epoch(self, run, dataloader, model):
        self.predict_aug_and_raw_scenes(dataloader, model)

        num_samples = len(dataloader) * run.num_gpus * run.samples_per_gpu
        assert len(self.raw_scene_preds_dict) == len(self.aug_scene_preds_dict) == num_samples
        assert sum([fid in self.raw_scene_preds_dict for fid in self.aug_scene_preds_dict]) == num_samples

        self.missing_annotated_mining(dataloader)

        self.aug_scene_preds_dict = {}
        self.raw_scene_preds_dict = {}
        run.to_epoch_bar.update(pseudo=self.ins_bank.num_pseudo_labels_in_infos) if hasattr(run, 'to_epoch_bar') else None


@Hook.priority(3)
class SS3DHook(SS3DHookHelper):
    def __init__(self):
        super().__init__()
        from ...api.config import Config
        self.cfg = Config.cfg.RUN.ss3d

        self.ins_bank = Bank(self.cfg.instance_bank, self.cfg.root_dir, self.cfg.class_names)
        self.miner = MineUnlabeledInstance(self.cfg.missing_anno_ins_mining, self.ins_bank)

    def run_begin(self, run, dataloaders, *args, **kwargs):
        def mine_miss_anno_ins_one_epoch(run, dataloader, *args, **kwargs):
            self.ss3d_one_iteration_begin(run, dataloader, *args, **kwargs)

        run.max_epochs += self.cfg.iter_num * self.cfg.epochs
        run.__class__.mine_miss_anno_ins_one_epoch = mine_miss_anno_ins_one_epoch

    def ss3d_one_iteration_begin(self, run, dataloader, *args, **kwargs):
        from ...runner.optimization import build_scheduler

        run.scheduler = build_scheduler(
            self.cfg.lr, optimizer=run.optimizer.optimizer,
            total_steps=len(dataloader) * run.num_gpus * self.cfg.epochs
        )

        self.miner.mine_unlabeled_instance_one_epoch(run, dataloader, *args, **kwargs)

    def train_one_epoch_begin(self, run, model, dataloader, *args, **kwargs):
        augmentor = dataloader.dataset.data_augmentor
        new_queue = self.new_aug_queue_with_pfa(run, augmentor)
        self.insert_points_filling = replace_attr(augmentor, 'data_augmentor_queue', new_queue).__enter__()

    def forward_end(self, result, run, batch_dict, *args, **kwargs):
        if run.state == 'train':
            mine_bg_cfg = run.ss3d.reliable_background_mining
            self.bg_holes_boxes.update(self.reliable_background_mining(batch_dict, run.model, mine_bg_cfg))

    def train_one_epoch_end(self, run, dataloader, *args, **kwargs):
        with barrier():
            all_gather_object(bg_holes_boxes=self.bg_holes_boxes)
            if run.rank == 0:
                assert len(self.bg_holes_boxes) == len(dataloader) * run.num_gpus * run.samples_per_gpu

                save_path = self.root_dir / run.ss3d.reliable_background_mining.fill_pts_info_path
                pickle.dump(self.bg_holes_boxes, open(save_path, 'wb'))
                num = sum([v.shape[0] for v in self.bg_holes_boxes.values()])
                print(f'bg_hole: {num / len(self.bg_holes_boxes)}')
            self.bg_holes_boxes = {}
            self.insert_points_filling.__exit__()
