import os
from pathlib import Path

import torch
import shutil
from collections import defaultdict
from ..api.dist import on_rank0, acc
from ..api import checkpoint, log
from ..utils.base import Hook


class CheckPointHookHelper:
    def __init__(self):
        self.metric_trackers = {}
        self.best_model = defaultdict(list)
        self.model_zoom = 'tools/models'
        self.logger = log.create_logger("ckpt helper", stderr=False)

    @staticmethod
    def try_load_checkpoint(run):
        if run.mode == 'train':
            if run.ckpt_list:
                if run.pretrain:
                    checkpoint.load_from_file(filename=run.ckpt_list[-1], model=run.model)
                else:
                    # restore cur_epochs & cur_iters
                    checkpoint.load_from_file(filename=run.ckpt_list[-1], model=run.model,
                                              optimizer=run.optimizer, scheduler=run.scheduler, runner=run)
            else:
                run.logger.info("==> train model from scratch")
        elif run.mode == 'test':
            assert run.ckpt_list
            run.max_epochs = len(run.ckpt_list)

    @staticmethod
    def upload_wandb(path, name, alias=None):
        if hasattr(acc, "trackers"):
            wandb = acc.get_tracker("wandb")
            model = wandb.Artifact(str(name), type='model')
            model.add_file(str(path))
            wandb.log_artifact(model, aliases=alias)
            wandb.save(str(path), policy='now')

    def init_metrics(self, run):
        for m in run.tracker.get('metrics', {}):
            direct = 1 if m.goal == 'maximize' else -1
            if m.get('save', False) and m.summary == "best":  # only metrics summarized in best rule are allow to save
                (run.ckpt_dir / f'{m.key}').mkdir(parents=True, exist_ok=True)
                self.metric_trackers[m.key] = (-1e10 * direct, -1, lambda o, n: direct * o < direct * n)

    def save_best_ckpt_if_metrics_update(self, run, eval_result):
        for key, (old, epoch, handler) in self.metric_trackers.items():
            new = eval_result[key]
            need_update = handler(old, new)
            if not need_update:
                continue
            self.metric_trackers[key] = (new, run.cur_epochs, handler)  # update metric

            """ get latest ckpt """
            if run.mode == 'train':
                ckpt = run.ckpt_list[-1]
            else:
                ckpt = run.ckpt_list[run.cur_epochs - 1]
            run.logger.info(f'save_best_ckpt_if_metrics_update: {ckpt} - {key}')

            """ how many epochs the ckpt was trained """
            try:
                epoch_mark = torch.load(ckpt)['runner_state']['cur_epochs']
            except:
                try:
                    epoch_mark = str(Path(ckpt).stem).split(checkpoint.CKPT_PATTERN)
                    if len(str(Path(ckpt).stem).split(checkpoint.CKPT_PATTERN)) == 2:
                        epoch_mark = int(epoch_mark[1])
                except:
                    epoch_mark = run.cur_epochs

            # remove older best model
            if self.best_model[key]:
                os.remove(self.best_model[key][-1])

            # save new best model
            filepath = run.ckpt_dir / key / f'{new}@{epoch_mark}.pth'
            self.best_model[key].append(filepath)
            shutil.copy(ckpt, self.best_model[key][-1])

            self.logger.info(f"************************"
                             f"save best checkpoint to {self.best_model[key][-1]}@{epoch_mark}"
                             f"************************")

    def save_best_ckpt_to_model_zoom(self, run):
        for key, file_list in self.best_model.items():
            if file_list:
                model_name = '_'.join([
                    f'{run.tags.model}',
                    f'{run.num_gpus}x{run.samples_per_gpu}',
                    f'{run.max_epochs}e',
                    run.tags.dataset,
                ])
                filepath = self.model_zoom / f'{model_name}({run.tags.experiment}).pth'
                shutil.copy(file_list[-1], filepath)
                # self.upload_wandb(filepath, f'{model_name}', alias=['latest', run.tags.experiment])

    def remove_older_ckpt(self, run):
        ckpt_list = checkpoint.potential_ckpts(None, run.ckpt_dir)
        if len(ckpt_list) >= run.checkpoints.max:
            for cur_file_idx in range(0, len(ckpt_list) - run.checkpoints.max + 1):
                os.remove(ckpt_list[cur_file_idx])
                self.logger.info(f"remove {ckpt_list[cur_file_idx]}")

    def save_this_ckpt(self, run):
        ckpt_name = run.ckpt_dir / f'{checkpoint.CKPT_PATTERN}{run.cur_epochs}.pth'
        checkpoint.save_to_file(filename=ckpt_name,
                                runner=run,
                                model=run.model,
                                optimizer=run.optimizer,
                                scheduler=run.scheduler)
        self.logger.info(f"save {ckpt_name.name}")
        run.ckpt_list.append(ckpt_name)


@Hook.priority(2)
class CheckPointHook(CheckPointHookHelper):
    def __init__(self):
        super().__init__()

    def run_begin(self, run, *args, **kwargs):
        self.try_load_checkpoint(run)
        self.model_zoom = Path(run.checkpoints.get('model_zoom', self.model_zoom))
        self.model_zoom.mkdir(exist_ok=True, parents=True)
        self.init_metrics(run)
        self.logger = log.create_logger(name="ckpt", log_file=run.ckpt_dir / "log.txt", stderr=False)

    def test_one_epoch_begin(self, run, *args, **kwargs):
        if not run.mode == "train":
            checkpoint.load_from_file(filename=run.ckpt_list[run.cur_epochs], model=run.model)

    @on_rank0
    def train_one_epoch_end(self, run, *args, **kwargs):
        if run.cur_epochs % run.checkpoints.interval == 0:
            self.remove_older_ckpt(run)
            self.save_this_ckpt(run)

    @on_rank0
    def test_one_epoch_end(self, run, *args, **kwargs):
        if len(run.ckpt_list) == 1:  # do not save best model for one ckpt testing.
            return
        if hasattr(run, 'from_eval') and 'eval_result' in run.from_eval:
            self.save_best_ckpt_if_metrics_update(run, run.from_eval['eval_result'])

        if run.cur_epochs == run.max_epochs:  # try to upload the best model to wandb Artifact model zoom
            self.save_best_ckpt_to_model_zoom(run)
