import _init_path
import os
import argparse
from pathlib import Path
from easydict import EasyDict

from rd3d.utils.base import Hook
from rd3d.api import acc, get_dist_state, set_random_seed, checkpoint, Config, create_logger
from rd3d import build_detector, build_dataloader, build_optimizer, build_scheduler, DistRunner


@Hook.auto
def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='specify the config for training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs='...', help='set extra config keys if needed')

    parser.add_argument('--experiment', type=str, default='default', help='experiment name')
    parser.add_argument('--output', type=str, default='output', help='experiment output root')
    parser.add_argument('--save_eval_label', action='store_true', default=False, help='')

    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrain', action='store_true', default=False, help='pretrained_model')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train for')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    return parser


@Hook.auto
def parse_config(args):
    """read config from file and cmdline"""
    cfg = Config.fromfile(args.cfg)
    cfg = Config.merge_custom_cmdline_setting(cfg, args.set_cfgs) if args.set_cfgs is not None else cfg
    """command-line inputs"""
    cfg.ARGS = vars(args)
    """runner state"""
    run_cfg = cfg.RUN
    mode = Path(__file__).stem
    dist_state = get_dist_state()

    tags = EasyDict(dataset=cfg.DATASET.TYPE,
                    model=cfg.config_file.stem,
                    experiment=args.experiment,
                    mode=cfg.DATASET.DATA_SPLIT[mode])

    run_cfg.update(mode=mode,
                   pretrain=args.pretrain,
                   rank=dist_state.process_index,
                   num_gpus=dist_state.num_processes,
                   max_epochs=args.epochs or run_cfg.max_epochs,
                   distributed=dist_state.distributed_type,
                   samples_per_gpu=args.batch_size or run_cfg.samples_per_gpu,
                   seed=args.seed if args.seed is not None else run_cfg.get('seed', 0))

    run_cfg.tags = tags
    run_cfg.seed = run_cfg.seed + run_cfg.rank
    run_cfg.save_eval_label = args.save_eval_label

    def mkdir():
        from datetime import datetime
        output_root = Path(args.output or cfg.project_root / 'output').absolute()
        run_cfg.output_dir = output_root / tags.dataset / tags.model / tags.experiment / tags.mode
        run_cfg.ckpt_dir = run_cfg.output_dir / 'ckpt'
        run_cfg.eval_dir = run_cfg.output_dir / 'eval'
        run_cfg.log_file = run_cfg.output_dir / f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        run_cfg.ckpt_list = checkpoint.potential_ckpts(args.ckpt, default=run_cfg.ckpt_dir)

        Path.mkdir(cfg.RUN.ckpt_dir, parents=True, exist_ok=True)
        Path.mkdir(cfg.RUN.eval_dir, parents=True, exist_ok=True)
        Path.mkdir(cfg.RUN.output_dir, parents=True, exist_ok=True)
        Path.mkdir(cfg.RUN.log_file.parent, parents=True, exist_ok=True)

    mkdir()
    return cfg


if __name__ == '__main__':
    """ init config """
    cfg = parse_config(add_args().parse_args())
    set_random_seed(cfg.RUN.seed)
    logger = create_logger(cfg.RUN.log_file)
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    logger.info(f"TOTAL_BATCH_SIZE: {cfg.RUN.num_gpus * cfg.RUN.samples_per_gpu}")
    logger.info(cfg.ARGS, title=["CMDLINE ARGS", "VALUE"])
    logger.info(cfg, title=[f'cfg.RUN', 'VALUE'], width=50)

    if cfg.RUN.tracker.get('init_kwargs', None):
        acc.init_trackers(cfg.RUN.tracker.project, config=cfg, init_kwargs=cfg.RUN.tracker.init_kwargs)

    """ build dataloaders """
    dataloaders = {'train': build_dataloader(cfg.DATASET, cfg.RUN, training=True, logger=logger)}
    if 'test' in [work.split for work in cfg.RUN.workflows[cfg.RUN.mode]]:
        dataloaders.update({'test': build_dataloader(cfg.DATASET, cfg.RUN, training=False, logger=logger)})

    """ & model & optimizer & lr_scheduler """
    model = build_detector(cfg.MODEL, dataset=dataloaders["train"].dataset)
    optim = build_optimizer(cfg.OPTIMIZATION, model=model)
    lr_sche = build_scheduler(cfg.LR, optimizer=optim, total_steps=len(dataloaders["train"]) * cfg.RUN.max_epochs)

    """ run experiment """
    runner = DistRunner(cfg.RUN, model=model, optimizer=optim, scheduler=lr_sche, logger=logger)
    runner.run(ckpts=cfg.RUN.ckpt_list, dataloaders=dataloaders)

    logger.info("Done")
