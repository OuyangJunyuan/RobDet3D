import torch
import easydict
import numpy as np
from ..utils.base import Hook
from ..api.dist import on_rank0


@Hook.priority()
class WandbCfgHook:
    @staticmethod
    def id_from_rundir(last_wandb_run_dir):
        last_wandb_cfg = last_wandb_run_dir / 'files' / 'config.yaml'
        import os
        import yaml
        if os.path.isfile(last_wandb_cfg):
            with open(last_wandb_cfg, 'r') as f:
                last_cfg = yaml.safe_load(f)
            try:
                return last_cfg['RUN']['value']['id']
            except KeyError:
                return None

    def add_args_end(self, parser):
        from accelerate.tracking import WandBTracker
        import wandb

        def store_init_configuration(self, values: dict):
            wandb.config.update(values, allow_val_change=True)

        WandBTracker.store_init_configuration = store_init_configuration

        subparser = parser.add_subparsers(dest='wandb', help='wandb args')
        wandb_subparser = subparser.add_parser('wandb', help='use wandb or not')
        wandb_subparser.add_argument('--project', type=str, default='RobDet3D')
        wandb_subparser.add_argument('--group', type=str, default=None, help='exp group name')
        wandb_subparser.add_argument('--id', type=str, default=None, help='unique id for a run')
        wandb_subparser.add_argument('--notes', type=str, default=None, help='like what specific to git -m')
        wandb_subparser.add_argument('--tags', type=str, default=[], nargs='+', help='tags to describe this exp')

    @on_rank0
    def parse_config_end(self, cfg, args):
        import time
        import wandb.util
        if args.wandb is not None:
            wandb_init_kwargs = easydict.EasyDict()
            training = cfg.RUN.mode == "train"
            run_id = self.id_from_rundir(cfg.RUN.output_dir / 'wandb' / 'latest-run') if args.id is None else args.id
            resume = run_id is not None and training

            if resume:
                wandb_init_kwargs.update(dir=cfg.RUN.output_dir, id=run_id, resume='must')
            else:
                run_id = wandb.util.generate_id()
                model = cfg.config_file.stem
                tags = [f'{k}[{v}]' for k, v in cfg.RUN.tags.items()] + [*args.tags]

                name = ' | '.join([model, args.experiment, time.strftime('%Y-%m-%d %H:%M:%S')])

                eval_prefix = '[eval]' if not training else ''
                group = f"{eval_prefix}{model}" if args.group is None else args.group

                wandb_init_kwargs.update(
                    dir=cfg.RUN.output_dir, allow_val_change=True,
                    group=group, name=name, id=run_id, notes=args.notes, tags=tags, resume='allow'
                )
            cfg.RUN.id = run_id

            init_kwargs = cfg.RUN.tracker.get('init_kwargs', easydict.EasyDict())
            init_kwargs.update(wandb=wandb_init_kwargs)
            cfg.RUN.tracker.update(project=args.project, init_kwargs=init_kwargs)


def __check_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif x is None:
        return x
    else:
        raise NotImplemented


def wandb_xyzrgb(xyz, rgb=None):
    if rgb is None:
        import matplotlib.colors as colors
        norm = np.linalg.norm(xyz, axis=1)
        min_val = np.min(norm)
        max_val = np.max(norm)
        norm = (norm - min_val) / (max_val - min_val)
        rgb = np.array([colors.hsv_to_rgb([n, 0.4, 0.5]) for n in norm]) * 255.0
    return np.concatenate([xyz, rgb], axis=1)


def wandb_boxes(boxes, label=None, score=None, color=None):
    from ..utils.box_utils import boxes_to_corners_3d
    corner3d = boxes_to_corners_3d(boxes)
    wandb_box_list = []
    for i in range(boxes.shape[0]):
        label_str = "unknown" if label is None else f'{label[i]}#{i}'
        label_str = label_str + ('' if score is None else "({:.1f})".format(score[i]))
        wandb_box_list.append({"corners": corner3d[i].tolist(),
                               "label": label_str,
                               "color": [255, 255, 255] if color is None else color})
    return wandb_box_list
