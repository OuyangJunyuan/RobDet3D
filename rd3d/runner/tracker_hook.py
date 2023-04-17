import numpy as np
from ..utils.base import Hook
from ..api.dist import acc, on_rank0


class ExpTrackerHookHelper:
    def __init__(self):
        self.freq_to_log = None
        self.metrics = []
        self.enable = False
        self.step_per_epoch = 2 << 30
        self.wandb = None

    def freq(self, name):
        i = self.freq_to_log.get(name, 0)
        if i < 0:
            i = self.step_per_epoch + i
        elif i == 0:
            i = 2 << 30
        elif 0 < i < 1:
            i = i * self.step_per_epoch
        return int(i)

    def need_log(self, name, step):
        return step % self.freq(name) == 0

    def log_train_info(self, run, info_dict, step):
        acc.log({'meta/epoch': run.cur_epochs,
                 'meta/lr': run.to_tracker['lr'],
                 'meta/loss': run.to_tracker['loss'],
                 'meta/grad_norm': run.to_tracker['grad'],
                 **{f"train/{key}": val for key, val in info_dict.items()}
                 }, step=step)

    def log_scene(self, run, pred_dict, batch_dict, step):
        import wandb
        from ..api.wandb import wandb_xyzrgb, wandb_boxes
        num_scenes = batch_dict['batch_size']
        frame_id = batch_dict['frame_id']
        points = batch_dict['points'][:, 1:4].reshape(num_scenes, -1, 3)[:, ::10, :]
        cls_map = ['unknown'] + run.model.class_names
        for i in range(num_scenes):
            numpify = lambda x: x.detach().cpu().numpy()
            scene_name = f"{frame_id[0]}.{run.cur_epochs}e"
            xyz = numpify(points[i])
            gt_boxes = numpify(batch_dict['gt_boxes'][i])
            gt_label = [cls_map[int(i)] for i in gt_boxes[:, -1]]
            pred_boxes = numpify(pred_dict[i]['pred_boxes'])
            pred_label = [cls_map[int(i)] for i in numpify(pred_dict[i]['pred_labels'])]
            scene_object_init_dict = {
                "type": "lidar/beta", "points": wandb_xyzrgb(xyz),
                "boxes": np.array(wandb_boxes(gt_boxes, label=gt_label, color=[0, 255, 0]) +
                                  wandb_boxes(pred_boxes, label=pred_label, color=[255, 255, 0]))
            }
            acc.log({f'scenes/{scene_name}': wandb.Object3D(scene_object_init_dict),
                     'meta/epoch': run.cur_epochs
                     }, step=step)
            break  # first scene in batch

    def log_metrics(self, run, **ld):
        metrics_list = [m.key for m in run.tracker.metrics]
        td = {f'test/{k}': v for k, v in ld.items()}
        ld = {f'metrics/{k}': v for k, v in ld.items() if k in metrics_list}
        acc.log({**td, **ld, 'meta/epoch': run.cur_epochs}, step=run.cur_iters)

    def log_param_and_grad(self, run):
        """model watch"""
        type_map = {'grad&param': 'all', 'grad': 'gradients', 'param': 'parameters'}
        for k in self.freq_to_log:
            if k in type_map:
                self.wandb.watch(models=run.model, log=type_map[k], log_freq=self.freq(k))

    def log_codes(self, cfg):
        include_fn = lambda path: bool(sum([str(path).endswith(p) for p in cfg.get('include', ['py'])]))
        self.wandb.log_code(root=cfg.root, include_fn=include_fn)

    def def_metrics(self, run, metrics_cfg):
        """set metrics with prefix 'test' to the axis-x meta/epoch"""
        self.wandb.define_metric('meta/epoch')
        self.wandb.define_metric('test/*', step_metric='meta/epoch')
        for kw in metrics_cfg:
            # metrics tracking, refer to https://docs.wandb.ai/ref/python/run#define_metric
            self.wandb.define_metric(f'metrics/{kw.key}', step_metric='meta/epoch', summary=kw.summary, goal=kw.goal)


@Hook.priority()
class ExpTrackerHook(ExpTrackerHookHelper):
    """
    before_run: 模型和梯度记录频率设置
    after_forward: 每次迭代后跟踪数据到对应step
    after_test_one_epoch： 每轮验证后从EvalHook读取指标并记录到对于epoch
    """
    ENABLE = True

    def __init__(self):
        super().__init__()

    @on_rank0
    def run_begin(self, run, *args, **kwargs):
        self.ENABLE = hasattr(acc, "trackers")
        run.to_tracker = {}

    @on_rank0
    def epoch_loop_begin(self, run, dataloaders, *args, **kwargs):
        self.wandb = acc.get_tracker('wandb')
        self.freq_to_log = run.tracker.interval
        self.step_per_epoch = len(dataloaders[run.state])

        if 'train' == run.mode: self.log_param_and_grad(run)
        if 'save_codes' in run.tracker: self.log_codes(run.tracker.save_codes)
        if 'metrics' in run.tracker: self.def_metrics(run, run.tracker.metrics)

    @on_rank0
    def batch_loop_begin(self, run, dataloader, *args, **kwargs):
        self.step_per_epoch = len(dataloader)

    @on_rank0
    def forward_end(self, result, run, batch_dict, *args, **kwargs):
        step = run.cur_iters
        pred_dict, info_dict = result

        if run.state == 'train' and self.need_log('train_info', run.cur_iters):
            self.log_train_info(run, info_dict, step)

        if run.state != 'train' and self.need_log('scenes', run.inner_iters + 1):
            self.log_scene(run, pred_dict, batch_dict, step)

    @on_rank0
    def test_one_epoch_end(self, run, *args, **kwargs):
        if 'metrics' in run.tracker:
            self.log_metrics(run, **run.to_tracker.get('recall', {}), **run.to_tracker.get('eval_result', {}))
