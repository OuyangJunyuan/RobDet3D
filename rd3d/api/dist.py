import distutils.sysconfig
import time

import torch.distributed as dist
from accelerate.accelerator import Accelerator
from accelerate.accelerator import AcceleratorState

acc = Accelerator(log_with=["wandb"])


def get_dist_state():
    if hasattr(get_dist_state, 'state'):
        return get_dist_state.state
    try:
        state = AcceleratorState()
        get_dist_state._state = state
    except:
        state = AcceleratorState(_from_accelerator=True)
    return state


def on_rank0(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_dist_state().process_index == 0:
            return func(*args, **kwargs)

    return wrapper


class barrier:
    def __init__(self):
        self.num_processes = get_dist_state().num_processes

    def __enter__(self):
        if self.num_processes > 1:
            dist.barrier()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.num_processes > 1:
            dist.barrier()

    def __call__(self, func):
        return func


def __collect_results_gpu(result_part, size):
    """
    copy from mmdetection3d, implement object gather by pickle, which like torch.all_gather_object
    """
    # import torch.distributed as dist
    # rank, world_size = get_dist_info()
    # # dump result part to tensor with pickle
    # part_tensor = torch.tensor(
    #     bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # # gather all result part tensor shape
    # shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    # shape_list = [shape_tensor.clone() for _ in range(world_size)]
    # dist.all_gather(shape_list, shape_tensor)
    # # padding result part tensor to max length
    # shape_max = torch.tensor(shape_list).max()
    # part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    # part_send[:shape_tensor[0]] = part_tensor
    # part_recv_list = [
    #     part_tensor.new_zeros(shape_max) for _ in range(world_size)
    # ]
    # # gather all result part
    # dist.all_gather(part_recv_list, part_send)
    #
    # if rank == 0:
    #     part_list = []
    #     for recv, shape in zip(part_recv_list, shape_list):
    #         part_list.append(
    #             pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
    #     # sort the results
    #     ordered_results = []
    #     for res in zip(*part_list):
    #         ordered_results.extend(list(res))
    #     # the dataloader may pad some samples
    #     ordered_results = ordered_results[:size]
    #     return ordered_results


def all_gather_object(**kwargs):
    """
    :param kwargs: object to be gathered in-place.
    """
    state = get_dist_state()
    if state.num_processes <= 1:
        return

    opt = kwargs.get('opt', None)
    for k, v in kwargs.items():
        output_list = [None] * state.num_processes
        dist.all_gather_object(output_list, v)
        if state.process_index == 0:
            if isinstance(v, list):
                if opt == 'merge':
                    kwargs[k] = sum([list(res) for res in zip(*output_list)])
                else:
                    kwargs[k] = sum(output_list)
            if isinstance(v, dict):
                for d in output_list:
                    kwargs[k].update(d)
