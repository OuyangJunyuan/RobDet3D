import torch
import pickle
import numpy as np
from tqdm import tqdm

from ...api import dist, log
from ...utils.base import Hook, replace_attr

from .instance_bank import InstanceBank
from .points_filling import InstanceFilling


class UnlabeledInstanceMiningModule:
    def __init__(self, cfg, ins_bank: InstanceBank):
        from ...datasets.augmentor.transforms import AugmentorList

        self.cfg = cfg
        self.ins_bank = ins_bank
        self.root_path = self.ins_bank.root_path / "ss3d/instance_mining"
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.logger = log.create_logger(name="ins", log_file=self.root_path / "log.txt", stderr=False)

        self.global_augments = AugmentorList(self.cfg.global_augments)

        self.aug_scene_preds_dict = {}
        self.raw_scene_preds_dict = {}

        self.logger.warning("*************** UnlabeledInstanceMiningModule ***************")
        self.logger.info(self.cfg)

    @staticmethod
    def load_data_to_numpy(args):
        for arg in args:
            for k in arg:
                arg[k] = arg[k].detach().cpu().numpy()
        return args

    @staticmethod
    def filter_out_data(arg, keep):
        keep_dict = {}
        remove_dict = {}
        remove = np.logical_not(keep)
        for key in arg.keys():
            keep_dict[key] = arg[key][keep]
            remove_dict[key] = arg[key][remove]
        return keep_dict, remove_dict

    @staticmethod
    def merge_dicts(dict1, dict2):
        assert dict1.keys() == dict2.keys()
        ret = {}
        for key in dict1:
            ret[key] = np.concatenate((dict1[key], dict2[key]), axis=0)
        return ret

    def load(self, root=None):
        raw_path = (root or self.root_path) / 'raw_scene_preds_dict.pkl'
        aug_path = (root or self.root_path) / 'aug_scene_preds_dict.pkl'
        ret = aug_path.exists() and raw_path.exists()
        if ret:
            self.raw_scene_preds_dict = pickle.load(open(raw_path, 'rb'))
            self.aug_scene_preds_dict = pickle.load(open(aug_path, 'rb'))
            self.logger.warning(f"load cache {raw_path.name} and {aug_path.name}")
        return ret

    @dist.on_rank0
    def save(self, root=None):
        raw_path = (root or self.root_path) / 'raw_scene_preds_dict.pkl'
        aug_path = (root or self.root_path) / 'aug_scene_preds_dict.pkl'
        pickle.dump(self.raw_scene_preds_dict, open(raw_path, 'wb'))
        pickle.dump(self.aug_scene_preds_dict, open(aug_path, 'wb'))
        self.logger.info(f"save cache {raw_path.name} and {aug_path.name}")

    def score_guided_suppression(self, *args, score_thr=0.9, field='pred_scores'):
        results = []
        for arg in args:
            keep = arg[field] > score_thr
            for key in arg.keys():
                arg[key] = arg[key][keep]
            results.append(keep)
            self.logger.info("score filter (%f): %d -> %d" % (score_thr, len(keep), keep.sum()))
        return results

    def iou3d_guided_suppression(self, preds1, preds2, iou_thr=0.9):
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

        boxes1, boxes2 = preds1['pred_boxes'], preds2['pred_boxes']
        n, m = boxes1.shape[0], boxes2.shape[0]

        if n == 0 or m == 0:
            keep = ind1 = ind2 = np.array([], dtype=int)
            pairs_iou = np.array([], dtype=np.float32)
        else:
            iou = boxes_bev_iou_cpu(boxes1, boxes2)
            ind1, ind2 = np.arange(n), np.argmax(iou, -1)
            pairs_iou = iou[ind1, ind2]
            self.logger.info(f"pairs iou3d: {pairs_iou}")
            keep = pairs_iou > iou_thr

        for (arg, ind) in [(preds1, ind1[keep]), (preds2, ind2[keep])]:
            for key in arg.keys():
                arg[key] = arg[key][ind]
            arg['iou3d'] = pairs_iou[keep]
        self.logger.info("iou3d filter (%f): %d %d -> %d" % (iou_thr, len(boxes1), len(boxes2), keep.sum()))

    def image_guided_suppression(self, raw_preds, aug_preds, image_labels, iou_thr):
        from ...utils import box_utils

        def homo(x):
            return np.concatenate((x, np.ones_like(x[..., :1])), axis=-1)

        def lidar_to_img(x, num_corners=4):
            x = homo(x) @ mat_lidar2cam.T
            x = x @ mat_cam2img.T
            x = x[..., :2] / x[..., 2:3]
            if num_corners == 8:
                return x
            elif num_corners == 4:
                return box_utils.corners3d_to_corners2d(x)
            else:
                raise NotImplementedError

        def corners3d_img_to_boxes2d_p1p2(corners3d_img):
            p1 = corners3d_img.min(axis=1)
            p2 = corners3d_img.max(axis=1)
            boxes2d_img = np.concatenate((p1, p2), axis=-1)
            return boxes2d_img

        def boxes2d_cwh_to_boxes2d_p1p2(boxes2d):
            c = boxes2d[..., 0:2]
            wh = boxes2d[..., 2:4]
            return np.concatenate((c - wh / 2, c + wh / 2), axis=-1)

        def iou2d(boxes1, boxes2):
            boxes1 = torch.from_numpy(boxes1)
            boxes2 = torch.from_numpy(boxes2)
            iou = box_utils.boxes_iou_normal(boxes1, boxes2).numpy()
            return iou

        masks_2d, boxes_2d, calib = image_labels
        mat_lidar2cam = calib['lidar2cam']
        mat_cam2img = calib['cam2img']
        raw_boxes = raw_preds['pred_boxes']
        aug_boxes = aug_preds['pred_boxes']
        h, w = masks_2d.shape[:2]
        raw_box3d_corners3d = box_utils.boxes3d_to_corners_3d(raw_boxes)
        raw_box3d_corners3d_img = lidar_to_img(raw_box3d_corners3d, num_corners=4)
        raw_box2d_p1p2 = corners3d_img_to_boxes2d_p1p2(raw_box3d_corners3d_img)
        raw_box2d_p1p2 = np.clip(raw_box2d_p1p2.reshape(-1, 2), a_min=(0, 0), a_max=(w, h)).reshape(-1, 4)

        aug_box3d_corners3d = box_utils.boxes3d_to_corners_3d(aug_boxes)
        aug_box3d_corners3d_img = lidar_to_img(aug_box3d_corners3d, num_corners=4)
        aug_box2d_p1p2 = corners3d_img_to_boxes2d_p1p2(aug_box3d_corners3d_img)
        aug_box2d_p1p2 = np.clip(aug_box2d_p1p2.reshape(-1, 2), a_min=(0, 0), a_max=(w, h)).reshape(-1, 4)

        boxes_2d_p1p2 = boxes2d_cwh_to_boxes2d_p1p2(boxes_2d)
        raw_iou2d = iou2d(raw_box2d_p1p2, boxes_2d_p1p2).max(axis=-1)
        aug_iou2d = iou2d(aug_box2d_p1p2, boxes_2d_p1p2).max(axis=-1)
        keep = np.logical_and(raw_iou2d > iou_thr, aug_iou2d > iou_thr)
        raw_preds['iou2d'] = raw_iou2d
        aug_preds['iou2d'] = aug_iou2d
        self.logger.info(f"raw preds iou2d iou: {raw_iou2d}")
        self.logger.info(f"aug preds iou2d iou: {aug_iou2d}")
        self.logger.info("iou2d filter (%f): %d %d -> %d" % (iou_thr, len(raw_boxes), len(aug_boxes), keep.sum()))
        self.viz_image_guided_suppression(masks_2d, boxes_2d, raw_box3d_corners3d_img, aug_box3d_corners3d_img)
        return keep

    def viz_image_guided_suppression(self, masks_2d, boxes_2d, raw_boxes_2d, aug_boxes_2d):
        if not self.cfg.get('visualize', False):
            return
        import cv2 as cv
        import matplotlib.pyplot as plt
        from ...utils import box_utils
        factor = 255 / max(1, masks_2d.max())
        masks_2d = (masks_2d * factor).astype(np.uint8)
        masks_2d = cv.cvtColor(masks_2d, cv.COLOR_GRAY2BGR)

        corners2d = box_utils.boxes2d_to_corners_2d(boxes_2d)
        corners2d_lines = box_utils.corners2d_to_lines(corners2d)
        for k, lines in enumerate(corners2d_lines.astype(int)):
            for line in lines:
                cv.line(masks_2d, line[0].tolist(), line[1], (0, 255, 0), 1)
            cv.putText(masks_2d, f"{k}", lines[2][0].tolist(),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        if raw_boxes_2d.shape[1] == 8:
            raw_corners_2d_lines = box_utils.corners3d_to_lines(raw_boxes_2d)
        else:
            raw_corners_2d_lines = box_utils.corners2d_to_lines(raw_boxes_2d)
        for k, lines in enumerate(raw_corners_2d_lines.astype(int)):
            for line in lines:
                cv.line(masks_2d, line[0].tolist(), line[1], (255, 0, 0), 1)
            cv.putText(masks_2d, f"{k}", lines[2][0].tolist(),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        if aug_boxes_2d.shape[1] == 8:
            aug_corners_2d_lines = box_utils.corners3d_to_lines(aug_boxes_2d)
        else:
            aug_corners_2d_lines = box_utils.corners2d_to_lines(aug_boxes_2d)
        for k, lines in enumerate(aug_corners_2d_lines.astype(int)):
            for line in lines:
                cv.line(masks_2d, line[0].tolist(), line[1], (0, 0, 255), 1)
            cv.putText(masks_2d, f"{k}", lines[2][0].tolist(),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        h, w = masks_2d.shape[:2]
        plt.figure(figsize=(int(w / 100), int(h / 100)))
        plt.axis('off')
        plt.imshow(masks_2d)
        plt.tight_layout()
        plt.show()
        plt.close()

    def viz_before_filter(self, fid, raw_points, raw_pred, aug_pred):
        if not self.cfg.get('visualize', False):
            return
        from ...utils.viz_utils import viz_scenes
        raw_boxes = raw_pred['pred_boxes']
        aug_boxes = aug_pred['pred_boxes']
        pts_bank, boxes_bank, gt_masks_bank = self.ins_bank.get_scene(fid)

        raw_color = np.ones_like(raw_boxes[:, :3]) * np.array([1, 1, 1])
        aug_color = np.ones_like(aug_boxes[:, :3]) * np.array([0, 0, 0])
        bank_color = np.ones_like(boxes_bank[:, :3]) * np.array([1, 0, 0])

        boxes = np.vstack((raw_boxes[:, :7], aug_boxes[:, :7], boxes_bank[:, :7]))
        colors = np.vstack((raw_color, aug_color, bank_color))
        viz_scenes((raw_points, (boxes, colors)), title='raw/aug paired prediction')

    def viz_after_filter(self, fid, raw_points, raw_pred):
        if not self.cfg.get('visualize', False):
            return
        from ...utils.viz_utils import viz_scenes
        viz_scenes((raw_points, raw_pred['pred_boxes']), title='mined instance')

    @staticmethod
    def get_reliable_boxes(raw_preds, aug_preds):
        """
        raw_preds: {pred_boxes, pred_scores, pred_labels, iou3d, iou2d}
        """
        from ...utils.common_utils import limit_period
        weights = [0.6, 0.4]
        fused_keys = ['pred_boxes', 'pred_scores', 'iou3d']
        ret = {k: v for k, v in raw_preds.items() if k not in fused_keys}
        raw_preds['pred_boxes'][:, 6] = limit_period(raw_preds['pred_boxes'][:, 6], offset=0.5, period=2 * np.pi)
        aug_preds['pred_boxes'][:, 6] = limit_period(aug_preds['pred_boxes'][:, 6], offset=0.5, period=2 * np.pi)
        for k in fused_keys:
            fused = raw_preds[k] * weights[0] + aug_preds[k] * weights[1]
            ret[k] = fused.astype(raw_preds[k].dtype)
        unreliable_heading = np.abs(aug_preds['pred_boxes'][:, 6] - raw_preds['pred_boxes'][:, 6]) > np.pi / 4
        ret['pred_boxes'][unreliable_heading, 6] = raw_preds['pred_boxes'][unreliable_heading, 6]
        return ret

    @dist.on_rank0
    def missing_annotated_mining(self, dataloader):
        self.logger.warning("mining annotated one epoch")

        pbar_kwargs = dict(desc='instance mining', leave=False, disable=not dist.is_rank0())
        for fid in tqdm(iterable=self.raw_scene_preds_dict, **pbar_kwargs):
            raw_preds, *no_logs = self.raw_scene_preds_dict[fid]
            aug_preds, aug_logs = self.aug_scene_preds_dict[fid]
            raw_points = getattr(dataloader.dataset, self.cfg.get_points_func)(fid)
            image_labels = dataloader.dataset.get_pseudo_instances(fid, return_calib=True)
            self.global_augments.invert({'gt_boxes': aug_preds['pred_boxes']}, aug_logs)

            self.viz_before_filter(fid, raw_points, raw_preds, aug_preds)
            self.score_guided_suppression(
                raw_preds, aug_preds, score_thr=self.cfg.score_threshold_low
            )
            self.iou3d_guided_suppression(
                raw_preds, aug_preds, iou_thr=self.cfg.iou3d_threshold_low
            )
            iou_2d_keep = self.image_guided_suppression(
                raw_preds, aug_preds, image_labels, iou_thr=self.cfg.iou2d_threshold
            )
            match_2d_raw_pred, unmatch_2d_raw_pred = self.filter_out_data(raw_preds, iou_2d_keep)
            match_2d_aug_pred, unmatch_2d_aug_pred = self.filter_out_data(aug_preds, iou_2d_keep)
            self.score_guided_suppression(
                unmatch_2d_raw_pred, unmatch_2d_aug_pred, score_thr=self.cfg.score_threshold_high
            )
            self.iou3d_guided_suppression(
                unmatch_2d_raw_pred, unmatch_2d_aug_pred, iou_thr=self.cfg.iou3d_threshold_high
            )
            raw_preds = self.merge_dicts(match_2d_raw_pred, unmatch_2d_raw_pred)
            aug_preds = self.merge_dicts(match_2d_aug_pred, unmatch_2d_aug_pred)
            reliable_pred = self.get_reliable_boxes(raw_preds, aug_preds)
            self.viz_after_filter(fid, raw_points, reliable_pred)

            self.ins_bank.try_insert(fid, raw_points, **reliable_pred)

            self.logger.info(f"frame {fid} reliable: {len(reliable_pred['pred_boxes'])}")
        self.ins_bank.save_to_disk()

    @torch.no_grad()
    def predict_aug_and_raw_scenes(self, dataloader, model):
        self.logger.warning("predicting scenes one epoch")
        model.eval()

        pbar_kwargs = dict(desc='raw scenes', leave=False, disable=not dist.is_rank0())
        with replace_attr(dataloader.dataset.data_augmentor, data_augmentor_queue=[]):
            for batch_dict in tqdm(iterable=dataloader, **pbar_kwargs):
                dataloader.dataset.load_data_to_gpu(batch_dict)
                pred_dicts, info_dict = model(batch_dict)
                self.raw_scene_preds_dict.update({
                    fid: (pred, None) for fid, pred in
                    zip(batch_dict['frame_id'], self.load_data_to_numpy(pred_dicts))
                })

        pbar_kwargs = dict(desc='aug scenes', leave=False, disable=not dist.is_rank0())
        with replace_attr(dataloader.dataset.data_augmentor, data_augmentor_queue=[self.global_augments]):
            for batch_dict in tqdm(iterable=dataloader, **pbar_kwargs):
                dataloader.dataset.load_data_to_gpu(batch_dict)
                pred_dicts, info_dict = model(batch_dict)
                self.aug_scene_preds_dict.update({
                    fid: (pred, logs) for fid, pred, logs in
                    zip(batch_dict['frame_id'], self.load_data_to_numpy(pred_dicts),
                        batch_dict['augment_logs'])
                })

        dist.all_gather_object(raw_scene_preds_dict=self.raw_scene_preds_dict,
                               aug_scene_preds_dict=self.aug_scene_preds_dict)

        if dist.is_rank0():
            raw_nums = np.array([len(pred[0]['pred_boxes'])
                                 for pred in self.raw_scene_preds_dict.values()])
            aug_nums = np.array([len(pred[0]['pred_boxes'])
                                 for pred in self.aug_scene_preds_dict.values()])
            self.logger.warning(f"average {raw_nums.mean()} predicted objects from raw scenes")
            self.logger.warning(f"average {aug_nums.mean()} predicted objects from aug scenes")

    def mine_unlabeled_instance_one_epoch(self, run, dataloader, model):
        if not (self.cfg.cache and self.load()):
            self.predict_aug_and_raw_scenes(dataloader, model)
            self.save()

        num_samples = len(dataloader) * run.num_gpus * run.samples_per_gpu
        assert len(self.raw_scene_preds_dict) == len(self.aug_scene_preds_dict) == num_samples
        assert sum([fid in self.raw_scene_preds_dict for fid in self.aug_scene_preds_dict]) == num_samples

        # if not self.cfg.cache:
        self.missing_annotated_mining(dataloader)

        self.aug_scene_preds_dict = {}
        self.raw_scene_preds_dict = {}


class ReliableBackGroundMiningModule:
    def __init__(self, cfg, ins_bank: InstanceBank):
        self.cfg = cfg
        self.ins_bank = ins_bank
        self.root_path = self.ins_bank.root_path / "ss3d/background_mining"
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.logger = log.create_logger(name="bg", log_file=self.root_path / "log.txt", stderr=False)

        self.redundant_predicted_boxes = {}

        self.logger.warning("*************** ReliableBackGroundMiningModule ***************")
        self.logger.info(self.cfg)

    def load(self, root=None):
        path = (root or self.root_path) / "background_infos.pkl"
        ret = path.exists()
        if ret:
            self.redundant_predicted_boxes = pickle.load(open(path, 'rb'))
            self.logger.warning(f"load cache {path.name}")
        return ret

    @dist.on_rank0
    def save(self, root=None):
        path = (root or self.root_path) / "background_infos.pkl"
        pickle.dump(self.redundant_predicted_boxes, open(path, 'wb'))
        self.logger.info(f"save cache {path.name}")

    @staticmethod
    def load_data_to_numpy(dicts):
        for d in dicts:
            for k in d:
                d[k] = d[k].detach().cpu().numpy()
        return dicts

    def weaken(self, raw_post_processing_cfg):
        import copy
        assert raw_post_processing_cfg.SCORE_THRESH != self.cfg.score_threshold
        weak_post_processing = copy.deepcopy(raw_post_processing_cfg)
        weak_post_processing.SCORE_THRESH = self.cfg.score_threshold
        weak_post_processing.NMS_CONFIG.NMS_THRESH = 1.0
        return weak_post_processing

    @torch.no_grad()
    def predict_without_nms_and_low_score_threshold(self, run, dataloader, model):
        self.logger.warning("predicting redundant boxes one epoch")

        model.eval()
        model_cfg = run.model.model_cfg
        with replace_attr(model_cfg, POST_PROCESSING=self.weaken(model_cfg.POST_PROCESSING)):
            with replace_attr(dataloader.dataset.data_augmentor, data_augmentor_queue=[]):
                pbar_kwargs = dict(desc='background mining', leave=False, disable=not dist.is_rank0())
                for batch_dict in tqdm(iterable=dataloader, **pbar_kwargs):
                    dataloader.dataset.load_data_to_gpu(batch_dict)
                    pred_dicts, info_dicts = model(batch_dict)

                    if self.cfg.get('visualize', False):
                        from ...utils.viz_utils import viz_scenes
                        bs = batch_dict['batch_size']
                        viz_points = batch_dict['points'].view(bs, -1, 5)[-1].clone()
                        viz_boxes = pred_dicts[-1]['pred_boxes'].clone()
                        viz_scenes((viz_points, viz_boxes), title='background mining')

                    self.load_data_to_numpy(pred_dicts)
                    pred_boxes_list = [d['pred_boxes'] for d in pred_dicts]
                    self.redundant_predicted_boxes.update({
                        fid: boxes[self.ins_bank.check_collision_free(fid, boxes)] for fid, boxes in
                        zip(batch_dict['frame_id'], pred_boxes_list)
                    })

        dist.all_gather_object(redundant_predicted_boxes=self.redundant_predicted_boxes)
        if dist.is_rank0():
            boxes_nums = np.array([len(pred) for pred in self.redundant_predicted_boxes.values()])
            self.logger.warning(f"average {boxes_nums.mean()} predicted boxes")

    def mine_reliable_background_one_epoch(self, run, dataloader, model):
        if not (self.cfg.cache and self.load()):
            self.predict_without_nms_and_low_score_threshold(run, dataloader, model)
            self.save()

        num_samples = len(dataloader) * run.num_gpus * run.samples_per_gpu
        assert len(self.redundant_predicted_boxes) == num_samples


@Hook.priority(3)
class SS3DHook:
    def __init__(self):
        from ...api.config import Config
        self.cfg = Config.cfg.RUN.ss3d
        self.enable = False

        self.ins_bank = InstanceBank(self.cfg.instance_bank, self.cfg.root_dir, self.cfg.class_names)
        self.ins_miner = UnlabeledInstanceMiningModule(self.cfg.unlabeled_instance_mining, self.ins_bank)
        self.bg_miner = ReliableBackGroundMiningModule(self.cfg.reliable_background_mining, self.ins_bank)
        self.instance_filling_augmentor = [InstanceFilling(**self.cfg.instance_filling,
                                                           instance_bank=self.ins_bank,
                                                           background_miner=self.bg_miner)]
        self.raw_augmentor_queue = None

    @property
    def ss3d_augmentor_queue(self):
        new_queue = self.instance_filling_augmentor + [ag for ag in self.raw_augmentor_queue
                                                       if type(ag).__name__ == 'DataBaseSampler']
        assert len(new_queue) == 2
        return new_queue

    def run_begin(self, run, *args, **kwargs):
        def mine_miss_anno_ins_one_epoch(run, *args, **kwargs):
            self.ss3d_one_iteration_begin(run, *args, **kwargs)

        run.max_epochs += self.cfg.iter_num * self.cfg.epochs
        run.__class__.mine_miss_anno_ins_one_epoch = mine_miss_anno_ins_one_epoch

    def ss3d_one_iteration_begin(self, run, dataloader, *args, **kwargs):
        from ...runner.optimization import build_scheduler

        self.enable = True
        run.scheduler = build_scheduler(
            self.cfg.lr, optimizer=run.optimizer.optimizer,
            total_steps=len(dataloader) * run.num_gpus * self.cfg.epochs
        )

        self.ins_miner.mine_unlabeled_instance_one_epoch(run, dataloader, *args, **kwargs)

    def train_one_epoch_begin(self, run, model, dataloader, *args, **kwargs):
        if not self.enable:
            return
        self.bg_miner.mine_reliable_background_one_epoch(run, dataloader, model)

        self.raw_augmentor_queue = dataloader.dataset.data_augmentor.data_augmentor_queue
        dataloader.dataset.data_augmentor.data_augmentor_queue = self.ss3d_augmentor_queue

    def train_one_epoch_end(self, run, dataloader, *args, **kwargs):
        if not self.enable:
            return
        dataloader.dataset.data_augmentor.data_augmentor_queue = self.raw_augmentor_queue
