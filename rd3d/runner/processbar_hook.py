import tqdm
from ..utils.base import Hook
from ..api.dist import on_rank0


class ProcessBarHookHelper:
    def __init__(self):
        self.epoch_bar = None
        self.batch_bar = None
        self.epoch_bar_remain_keys = ['metric']

    def update_bar_display(self, run):
        self.batch_bar.set_postfix(total_it=run.cur_iters, **run.to_batch_bar)
        self.epoch_bar.set_postfix(**run.to_epoch_bar)

    def clear_bar_display(self, run):
        run.batch_bar = {}
        run.to_epoch_bar = {k: v for k, v in run.to_epoch_bar.items() if k in self.epoch_bar_remain_keys}


@Hook.priority()
class ProcessBarHook(ProcessBarHookHelper):
    ENABLE = True

    def __init__(self):
        super().__init__()

    @on_rank0
    def epoch_loop_begin(self, run, *args, **kwargs):
        run.to_epoch_bar = {}
        self.epoch_bar = tqdm.tqdm(total=run.max_epochs, initial=run.cur_epochs,
                                   leave=True, desc='epochs', dynamic_ncols=True)

    @on_rank0
    def batch_loop_begin(self, run, dataloader, *args, **kwargs):
        run.to_batch_bar = {}
        self.batch_bar = tqdm.tqdm(total=len(dataloader), desc=run.state,
                                   leave=run.cur_epochs + 1 == run.max_epochs, dynamic_ncols=True)

    @on_rank0
    def forward_end(self, result, run, *args, **kwargs):
        self.batch_bar.update()
        self.update_bar_display(run)

    @on_rank0
    def batch_loop_end(self, run, *args, **kwargs):
        self.batch_bar.close()
        if run.mode == run.state:
            self.epoch_bar.update()
        self.clear_bar_display(run)

    @on_rank0
    def epoch_loop_end(self, run, *args, **kwargs):
        self.epoch_bar.close()
