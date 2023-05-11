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

    @dist.on_rank0
    def missing_annotated_mining(self, dataloader):
        self.logger.warning("mining annotated one epoch")

        pbar_kwargs = dict(desc='instance mining', leave=False, disable=not dist.is_rank0())
        for fid in tqdm(iterable=self.raw_scene_preds_dict, **pbar_kwargs):
            raw_pred, *no_logs = self.raw_scene_preds_dict[fid]
            aug_pred, aug_logs = self.aug_scene_preds_dict[fid]
            raw_points = getattr(dataloader.dataset, self.cfg.get_points_func)(fid)

            self.global_augments.invert({'gt_boxes': aug_pred['pred_boxes']}, aug_logs)

            if self.cfg.get('visualize', False):
                from ...utils.viz_utils import viz_scenes
                pts_bank, boxes_bank, gt_masks_bank = self.ins_bank.get_scene(fid)
                viz_scenes((raw_points, raw_pred['pred_boxes']),
                           (raw_points, aug_pred['pred_boxes']),
                           (raw_points, boxes_bank),
                           offset=[0, 50, 0], title='raw/aug paired prediction')

            self.score_based_filter(raw_pred, aug_pred, score_thr=self.cfg.score_threshold_high)
            self.iou_guided_suppression(raw_pred, aug_pred, iou_thr=self.cfg.iou_threshold)
            reliable_pred = raw_pred

            if reliable_pred['pred_boxes'].shape[0] > 0:
                self.ins_bank.try_insert(fid, raw_points, **reliable_pred)

            if self.cfg.get('visualize', False):
                from ...utils.viz_utils import viz_scenes
                viz_scenes((raw_points, reliable_pred['pred_boxes']), title='mined instance')

            self.logger.info(f"frame {fid} "
                             f"raw({len(raw_pred['pred_boxes'])}) "
                             f"aug({len(aug_pred['pred_boxes'])}) "
                             f"-> reliable({len(reliable_pred['pred_boxes'])})")

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

        if not self.cfg.cache:
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
        super().__init__()
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
