import pickle

import numba
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from .instance_bank import Bank
from .points_filling import PointCloudFilling
from ...api.dist import on_rank0, all_gather_object, barrier
from ...datasets.augmentor.transforms import AugmentorList
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

    @staticmethod
    def score_based_filter(*args, score_thr=0.9, field='pred_scores'):
        for arg in args:
            keep = arg[field] > score_thr
            for key in arg.keys():
                arg[key] = arg[key][keep]

    @staticmethod
    def iou_guided_suppression(preds1, preds2, iou_thr=0.9, score_thr_low=0.2, score_thr_high=0.4):
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
        def bev_iou_cpu(box1, box2):
            n1, n2 = box1.shape[0], box2.shape[0]
            return boxes_bev_iou_cpu(box1, box2) if n1 and n2 else np.zeros([n1, n2])

        def argmax(x: np.ndarray, axis=-1):
            if x.shape[-1] == 0:
                return np.zeros([0], dtype=int)
            else:
                return np.argmax(x, axis=axis)

        """
        raw_pred: N
        aug_pred: M
        """
        iou = bev_iou_cpu(preds1['pred_boxes'], preds2['pred_boxes'])  # (N,M)
        n, m = iou.shape
        if n == 0 or m == 0:
            ind1 = ind2 = np.zeros([0], dtype=int)
        else:
            ind1, ind2 = np.arange(n), argmax(iou, -1)
        pairs_iou = iou[ind1, ind2]
        mask = pairs_iou > iou_thr
        ind1, ind2 = ind1[mask], ind2[mask]

        output = {}
        for k in preds1.keys():
            output[k] = []
            for i1, i2 in zip(ind1, ind2):
                s1, s2 = preds1['pred_scores'][i1], preds2['pred_scores'][i2]
                if s1 < score_thr_high and s2 < score_thr_high: continue
                preds, i = preds1, i1
                if s1 < s2: preds, i = preds2, i2
                output[k].append(preds[k][i])
        output = {k: np.array(v) for k, v in output.items()}
        return output
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

    @on_rank0
    def missing_annotated_mining(self, mine_ins_cfg, dataloader):

        get_points = getattr(dataloader.dataset, mine_ins_cfg.get_points_func)
        for fid in tqdm(iterable=self.aug_scene_preds_dict, desc='mining(filter)', leave=False, dynamic_ncols=True):
            raw_pred, *_ = self.raw_scene_preds_dict[fid]
            aug_pred, aug_logs = self.aug_scene_preds_dict[fid]
            raw_points = get_points(fid)

            self.global_augments.invert({'gt_boxes': aug_pred['pred_boxes']}, aug_logs)

            if mine_ins_cfg.get('visualize', False):
                from ...utils.viz_utils import viz_scenes
                print(raw_pred['pred_scores'])
                print(aug_pred['pred_scores'])
                bk = self.instance_bank.get_scene(fid)
                viz_scenes((raw_points, raw_pred['pred_boxes']),
                           (raw_points, aug_pred['pred_boxes']),
                           (raw_points, bk[1]), offset=[0, 50, 0])

            self.score_based_filter(raw_pred, aug_pred, score_thr=mine_ins_cfg.score_threshold_low)

            output_pred = self.iou_guided_suppression(raw_pred, aug_pred,
                                                      iou_thr=mine_ins_cfg.iou_threshold,
                                                      score_thr_low=mine_ins_cfg.score_threshold_low,
                                                      score_thr_high=mine_ins_cfg.score_threshold_high)

            if mine_ins_cfg.get('visualize', False):
                from ...utils.viz_utils import viz_scenes
                viz_scenes((raw_points, output_pred['pred_boxes']))

            if output_pred['pred_boxes'].shape[0] > 0:
                self.instance_bank.try_insert(fid, raw_points, **output_pred)

        self.instance_bank.save_to_file()

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


@Hook.priority(3)
class SS3DHook(SS3DHookHelper):
    from ..runner import DistRunnerBase

    def run_begin(self, run, dataloaders, *args, **kwargs):
        dataset = dataloaders['train'].dataset
        self.root_dir = dataset.root_path
        self.class_names = dataset.class_names

        self.global_augments = AugmentorList(run.ss3d.global_augments)
        self.instance_bank = Bank(run.ss3d.instance_bank, root_dir=self.root_dir, class_names=self.class_names)

        run.from_ss3d = self
        run.max_epochs += run.ss3d.iter_num * run.ss3d.epochs

    def forward_end(self, result, run, batch_dict, *args, **kwargs):
        if run.state != 'train':
            self.mine_miss_anno_ins_forward_end(result, run, batch_dict, *args, **kwargs)
        else:
            mine_bg_cfg = run.ss3d.reliable_background_mining
            self.bg_holes_boxes.update(self.reliable_background_mining(batch_dict, run.model, mine_bg_cfg))

    def mine_miss_anno_ins_one_epoch_begin(self, run, model, dataloader):
        from ...runner.optimization import build_scheduler
        self.num_samples = len(dataloader) * run.num_gpus * run.samples_per_gpu
        self.num_steps = len(dataloader) * run.num_gpus * run.ss3d.epochs
        run.scheduler = build_scheduler(run.ss3d.lr, optimizer=run.optimizer.optimizer, total_steps=self.num_steps)

    @Hook.auto_call
    @torch.no_grad()
    def mine_miss_anno_ins_one_epoch(run: DistRunnerBase, dataloader, *args, **kwargs):
        if run.ss3d.missing_anno_ins_mining.cache:  # skip mining when cache is used.
            return

        run.model.eval()
        augmentor = dataloader.dataset.data_augmentor

        with replace_attr(run, 'state', 'mine_aug_scene'):
            with replace_attr(augmentor, 'data_augmentor_queue', [run.from_ss3d.global_augments]):
                run.batch_loop(dataloader=dataloader, *args, **kwargs)

        with replace_attr(run, 'state', 'mine_raw_scene'):
            with replace_attr(augmentor, 'data_augmentor_queue', []):
                run.batch_loop(dataloader=dataloader, *args, **kwargs)

    DistRunnerBase.mine_miss_anno_ins_one_epoch = mine_miss_anno_ins_one_epoch

    def mine_miss_anno_ins_forward_end(self, result, run, batch_dict, *args, **kwargs):
        pred_dict, info_dict = result
        if run.state == 'mine_aug_scene':
            self.aug_scene_preds_dict.update({
                fid: (pred, logs) for fid, pred, logs in
                zip(batch_dict['frame_id'], self.load_data_to_numpy(pred_dict),
                    batch_dict['augment_logs'])
            })

        if run.state == 'mine_raw_scene':
            self.raw_scene_preds_dict.update({
                fid: (pred, None) for fid, pred in
                zip(batch_dict['frame_id'], self.load_data_to_numpy(pred_dict))
            })

    @barrier()
    def mine_miss_anno_ins_one_epoch_end(self, run, dataloader, *args, **kwargs):
        all_gather_object(raw_scene_preds_dict=self.raw_scene_preds_dict,
                          aug_scene_preds_dict=self.aug_scene_preds_dict)

        if run.rank == 0:
            if run.ss3d.missing_anno_ins_mining.cache:
                self.aug_scene_preds_dict = pickle.load(open(self.root_dir / 'ss3d/aug_scene_preds_dict.pkl', 'rb'))
                self.raw_scene_preds_dict = pickle.load(open(self.root_dir / 'ss3d/raw_scene_preds_dict.pkl', 'rb'))
            else:
                pickle.dump(self.aug_scene_preds_dict, open(self.root_dir / 'ss3d/aug_scene_preds_dict.pkl', 'wb'))
                pickle.dump(self.raw_scene_preds_dict, open(self.root_dir / 'ss3d/raw_scene_preds_dict.pkl', 'wb'))

            assert len(self.raw_scene_preds_dict) == len(self.aug_scene_preds_dict) == self.num_samples
            assert sum([fid in self.raw_scene_preds_dict for fid in self.aug_scene_preds_dict]) == self.num_samples

            self.missing_annotated_mining(run.ss3d.missing_anno_ins_mining, dataloader)

        self.aug_scene_preds_dict = {}
        self.raw_scene_preds_dict = {}
        run.to_epoch_bar.update(pseudo=self.instance_bank.num_pd_in_bk) if hasattr(run, 'to_epoch_bar') else None

    def train_one_epoch_begin(self, run, model, dataloader, *args, **kwargs):
        if self.enable:
            augmentor = dataloader.dataset.data_augmentor
            new_queue = self.new_aug_queue_with_pfa(run, augmentor)
            self.insert_points_filling = replace_attr(augmentor, 'data_augmentor_queue', new_queue).__enter__()

    def train_one_epoch_end(self, run, dataloader, *args, **kwargs):
        if self.enable:
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
