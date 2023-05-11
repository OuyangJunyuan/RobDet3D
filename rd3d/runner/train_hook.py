from ..api import acc
from ..utils.base import Hook
from torch.nn.utils import clip_grad_norm_


@Hook.auto_call
def train_one_epoch(run, *args, **kwargs):
    kwargs.get('model', run.model).train()
    run.batch_loop(*args, **kwargs)


class TrainHookHelper:

    @staticmethod
    def send_infos_to_display(run, **infos):
        if hasattr(run, 'to_epoch_bar'):
            run.to_epoch_bar.update(**{k: '%.2e' % v for k, v in infos.items()})
        if hasattr(run, 'to_tracker'):
            run.to_tracker.update(**infos)


@Hook.priority(1)
class TrainHook(TrainHookHelper):
    """
    forward_end: optimize model once and record loss,lr,norm
    """
    from .runner import DistRunnerBase
    ENABLE = True
    DistRunnerBase.train_one_epoch = train_one_epoch

    def run_begin(self, run, *args, **kwargs):
        TrainHook.ENABLE = run.mode == 'train'

    def forward_end(self, result, run, *args, **kwargs):
        if run.state != 'train': return
        ret_dict, info_dict = result
        loss = ret_dict['loss']
        lr = run.scheduler.get_last_lr()
        run.optimizer.zero_grad()
        acc.backward(loss)
        grad_norm = clip_grad_norm_(run.model.parameters(), run.grad_norm_clip)
        run.optimizer.step()
        run.scheduler.step()

        self.send_infos_to_display(run, lr=lr[0], grad=grad_norm, loss=loss.item())
