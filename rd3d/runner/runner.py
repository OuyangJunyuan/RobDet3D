import torch
import importlib
from ..api import acc
from ..api.log import create_logger, log_to_file

from ..utils.base import Hook


class DistRunnerBase:
    state_keys = ['cur_loops', 'cur_epochs', 'cur_iters', 'metrics', 'state']

    def __init__(self, run_cfg, model, optimizer=None, scheduler=None, logger=None):
        self.update(run_cfg)
        self.state = self.mode
        self.cur_loops, self.cur_epochs, self.cur_iters, self.inner_iters = 0, 0, 0, 0
        self.dist_model, self.dist_optim, self.dist_sche = acc.prepare(model, optimizer, scheduler)
        self.logger = logger if logger is not None else create_logger()

        if self.mode == "train":
            from . import train_hook, eval_hook
        else:
            from . import eval_hook
        from . import processbar_hook, tracker_hook, checkpoint_hook
        for m in getattr(self, 'custom_import', []): importlib.import_module(m)

    @staticmethod
    def member(func):
        setattr(DistRunnerBase, func.__name__, func)
        return func

    @property
    def training(self):
        return self.state == 'train'

    @property
    def handle_one_epoch(self):
        return getattr(self, f'{self.state}_one_epoch')

    @property
    def model(self):
        return acc.unwrap_model(self.dist_model)

    @property
    def optimizer(self):
        return self.dist_optim

    @property
    def scheduler(self):
        return self.dist_sche

    @scheduler.setter
    def scheduler(self, sche):
        self.dist_sche = acc.prepare(sche)

    def update(self, d={}, **kwargs):
        self.__dict__.update(d, **kwargs)

    def state_dict(self):
        return {k: getattr(self, k, None) for k in self.state_keys}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items(): setattr(self, k, v)


class DistRunner(DistRunnerBase):

    def __init__(self, run_cfg, model, optimizer=None, scheduler=None, logger=None):
        super().__init__(run_cfg, model, optimizer, scheduler, logger)

    @Hook.auto_call
    def forward(self, model, batch_dict):
        pred_ret_dict, ext_info_dict = model(batch_dict)
        if self.mode == self.state:
            self.cur_iters += 1
        return pred_ret_dict, ext_info_dict

    @Hook.auto_call
    def batch_loop(self, model, dataloader):
        # self.logger.info(f"********* Start EPOCH {self.cur_epochs} ({self.state}) *********")
        for self.inner_iters, batch_dict in enumerate(dataloader):
            dataloader.dataset.load_data_to_gpu(batch_dict)
            self.forward(model=model, batch_dict=batch_dict)
        if self.mode == self.state:
            self.cur_epochs += 1

    @Hook.auto_call
    def epoch_loop(self, dataloaders):
        cur_loops = 0
        while self.cur_epochs < self.max_epochs:
            for i, work in enumerate(self.workflows[self.mode]):
                self.update(work)
                for j in range(self.epochs):
                    cur_loops += 1
                    if cur_loops > self.cur_loops:
                        self.cur_loops = cur_loops
                        self.handle_one_epoch(model=self.dist_model, dataloader=dataloaders[self.split])
                    else:
                        self.logger.info(f'skip work({work})')

    @Hook.auto_call
    def run(self, ckpts, dataloaders):
        self.logger.info(f"*********{self.mode} {self.tags.dataset}/{self.tags.model}/{self.tags.experiment}*********")
        dataloaders = {k: acc.prepare(v) for k, v in dataloaders.items()}
        # with log_to_file():
        self.epoch_loop(dataloaders)
