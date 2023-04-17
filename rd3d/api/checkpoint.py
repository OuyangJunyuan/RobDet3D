import os
import torch
from ..api import create_logger

CKPT_PATTERN = 'checkpoint_epoch_'


def load_from_file(filename, model=None, optimizer=None, scheduler=None, runner=None):
    def _load_state_dict(model, model_state_disk, *, strict=True):
        from typing import Set
        try:
            import spconv.pytorch as spconv
        except:
            import spconv as spconv

        import torch.nn as nn
        def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
            """
            Finds all spconv keys that need to have weight's transposed
            """
            found_keys: Set[str] = set()
            for name, child in model.named_children():
                new_prefix = f"{prefix}.{name}" if prefix != "" else name

                if isinstance(child, spconv.conv.SparseConvolution):
                    new_prefix = f"{new_prefix}.weight"
                    found_keys.add(new_prefix)

                found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

            return found_keys

        state_dict = model.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(model)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            model.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            model.load_state_dict(state_dict)
        return state_dict, update_model_state

    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger = create_logger()
    device = next(model.parameters()).device
    logger.info(f'==> Loading checkpoint {filename} to {device}')
    checkpoint = torch.load(filename, map_location=device)
    logger.info(f'==> Checkpoint trained from version: {checkpoint.get("version", None)}')

    if model is not None:
        model_state = checkpoint['model_state']
        state_dict, update_model_state = _load_state_dict(model, model_state, strict=False)

        missing_dict = {key: state_dict[key].shape for key in state_dict if key not in update_model_state}
        unexpected_dict = {key: model_state[key].shape for key in model_state if key not in state_dict}

        logger.info(missing_dict, title=['Missing Key', 'Shape']) if missing_dict else None
        logger.info(unexpected_dict, title=['Unexpected Key', 'Shape']) if unexpected_dict else None
        logger.info(f'==> Loaded params for model ({len(update_model_state)}/{len(state_dict)})')
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        try:
            num_steps = scheduler.last_epoch
        except AttributeError:
            num_steps = scheduler.scheduler.last_epoch
        logger.info(f'==> Loaded params for scheduler @{num_steps}')

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f'==> Loaded params for optimizer')

    if runner is not None:
        runner.load_state_dict(checkpoint['runner_state'])
        logger.info(f'==> Loaded runner:')
        logger.info(checkpoint['runner_state'])


def save_to_file(model=None, optimizer=None, scheduler=None, runner=None, filename='checkpoint'):
    def model_state_to_cpu(model_state):
        model_state_cpu = type(model_state)()  # ordered dict
        for key, val in model_state.items():
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    try:
        import rd3d
        version = rd3d.__version__
    except:
        version = 'none'

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    runner_state = runner.state_dict() if runner is not None else {}
    state = dict(
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        runner_state=runner_state,
        version=version,
    )
    torch.save(state, filename)


def potential_ckpts(ckpt, default=None):
    import glob
    from pathlib import Path
    if ckpt is not None:
        if Path(ckpt).is_dir():
            default = ckpt
        else:
            return [ckpt]
    assert not (ckpt is None and default is None)
    ckpt_list = glob.glob(str(default / f'*{CKPT_PATTERN}*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    return ckpt_list
