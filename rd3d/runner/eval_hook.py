import time
import torch
import pickle
from collections import defaultdict

from ..api import acc
from ..utils.base import Hook
from ..api.dist import on_rank0, get_dist_state


@Hook.auto_call
@torch.no_grad()
def test_one_epoch(run, *args, **kwargs):
    kwargs.get('model', run.model).eval()
    run.batch_loop(*args, **kwargs)


class EvalHookHelper:
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    def __init__(self):
        self.dataset = None
        self.eval_metric_type = None
        self.recall_thresh = None
        self.output_dir = None
        self.output_label_dir = None

        self.pred_labels = []
        self.recalls = {}

        self.start_time = 0
        self.end_time = 0

    def gather_pred_labels_from_all_process(self):
        import torch
        import torch.distributed as dist
        if acc.distributed_type != "NO":
            num_process = get_dist_state().num_processes
            all_pred_labels = [_ for _ in range(num_process)]
            dist.all_gather_object(all_pred_labels, self.pred_labels)
            self.pred_labels = sum([list(res) for res in zip(*all_pred_labels)], [])

            for k, v in self.recalls.items():
                v = torch.tensor(v).cuda()
                dist.all_reduce(v)
                self.recalls[k] = v.item()

    @on_rank0
    def flatten_pred_labels(self):
        self.pred_labels = sum(self.pred_labels, [])
        # avoid over sample due to mismatch between len(dataset) and batch_size*num_gpus
        self.pred_labels = self.pred_labels[:len(self.dataset)]
        # for i in range(len(self.pred_labels)):
        #     if self.pred_labels[i]['frame_id'] != self.dataset.infos[i]['point_cloud']['lidar_idx']:
        #         raise ValueError

    def collect_recalls_from(self, info_dict):
        self.recalls['gt_num'] += info_dict.get('gt', 0)
        for k, v in info_dict.items():
            if k.startswith(('roi', 'rcnn')):
                self.recalls[f'recall_{k}'] += v

    def send_recalls_to_display(self, run):
        min_iou = self.recall_thresh[0]
        ss = f"({int(self.recalls[f'recall_roi_{min_iou}'])},{int(self.recalls[f'recall_rcnn_{min_iou}'])})" \
             f"/{int(self.recalls['gt_num'])}"
        epoch_bar_dict = {f'recall_{min_iou}': ss}

        if hasattr(run, 'to_epoch_bar'):
            run.to_epoch_bar.update(epoch_bar_dict)
        if hasattr(run, 'to_tracker'):
            run.to_tracker.update(epoch_bar_dict)

    def send_infos_to_display(self, run, result_dict):
        if hasattr(run, 'to_tracker'):
            run.to_tracker.update(recall=self.recalls, eval_result=result_dict)
        if hasattr(run, 'to_epoch_bar') and run.tracker.metrics:
            run.to_epoch_bar.update(metric=result_dict[run.tracker.metrics[0].key])


@Hook.priority(1)
class EvalHook(EvalHookHelper):
    from .runner import DistRunnerBase
    ENABLE = True
    DistRunnerBase.test_one_epoch = test_one_epoch

    def run_begin(self, run, *args, **kwargs):
        self.ENABLE = 'test' in [work.split for work in run.workflows[run.mode]]
        self.eval_metric_type = run.model.model_cfg.POST_PROCESSING.EVAL_METRIC
        self.recall_thresh = run.model.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST

    def test_one_epoch_begin(self, run, dataloader, *args, **kwargs):
        run.from_eval = {}
        self.pred_labels = []
        self.recalls = defaultdict(float)
        self.dataset = dataloader.dataset
        self.output_dir = run.eval_dir / f'epoch_{run.cur_epochs + 1}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_label_dir = self.output_dir / 'final_result'
        self.output_label_dir.mkdir(parents=True, exist_ok=True)

    def batch_loop_begin(self, *args, **kwargs):
        self.start_time = time.time()

    def forward_end(self, result, run, batch_dict, *args, **kwargs):
        if run.state == 'test':
            pred_dicts, info_dict = result
            pred_labels_batch = self.dataset.generate_prediction_dicts(batch_dict, pred_dicts,
                                                                       self.dataset.class_names,
                                                                       output_path=self.output_label_dir)

            self.pred_labels.append(pred_labels_batch)
            self.collect_recalls_from(info_dict)
            self.send_recalls_to_display(run)

    def batch_loop_end(self, run, *args, **kwargs):
        self.end_time = time.time()
        if run.state == 'test':
            self.gather_pred_labels_from_all_process()
            self.flatten_pred_labels()

    @on_rank0
    def test_one_epoch_end(self, run, *args, **kwargs):
        """ save pred_labels and logging recall information """
        pickle.dump(self.pred_labels, open(self.output_dir / 'result.pkl', 'wb'))
        # run.logger.info(f'Result is save to {self.output_dir}')

        num_samples = len(self.pred_labels)
        avg_pred_objs = sum([a['name'].__len__() for a in self.pred_labels]) / max(1, num_samples)
        run.logger.info('Average predicted number of objects(%d samples): %.3f' % (num_samples, avg_pred_objs))

        sec_per_example = (self.end_time - self.start_time) / len(self.dataset)
        run.logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        """ calculate recall from accumulated self.recalls """
        for k, n in self.recalls.items():
            if k.startswith('recall'):
                self.recalls[k] = n / max(self.recalls['gt_num'], 1)
                run.logger.info(f"{k}: {self.recalls[k]}")

        """ compute evaluation result and save to info_dict for experiments tracking """
        result_str, result_dict = self.dataset.evaluation(
            self.pred_labels, self.dataset.class_names,
            eval_metric=self.eval_metric_type,
            output_path=self.output_label_dir
        )
        run.logger.info(result_str)

        run.from_eval.update(recall=self.recalls, eval_result=result_dict)
        self.send_infos_to_display(run, result_dict)
