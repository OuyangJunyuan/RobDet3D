import os
import argparse
from pathlib import Path
from easydict import EasyDict

from rd3d.utils.base import Hook
from rd3d import build_detector, build_dataloader, DistRunner
from rd3d.api import acc, get_dist_state, set_random_seed, checkpoint, Config,create_logger


@Hook.auto_call
def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='output', help='experiment output root')
    parser.add_argument('--experiment', type=str, default='default', help='experiment name')
    parser.add_argument('--eval_tag', type=str, default='default', help='experiment name')
    parser.add_argument('--cfg', type=str, required=True, help='specify the config for training')

    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to start from')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--save_eval_label', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs='...', help='set extra config keys if needed')
    return parser


@Hook.auto_call
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

    tags = EasyDict(eval=args.eval_tag,
                    dataset=cfg.DATASET.TYPE,
                    model=cfg.config_file.stem,
                    experiment=args.experiment,
                    mode=cfg.DATASET.DATA_SPLIT[mode])

    run_cfg.update(mode=mode,
                   rank=dist_state.process_index,
                   num_gpus=dist_state.num_processes,
                   distributed=dist_state.distributed_type,
                   samples_per_gpu=args.batch_size or run_cfg.samples_per_gpu,
                   seed=args.seed if args.seed is not None else run_cfg.get('seed', 0))

    run_cfg.tags = tags
    run_cfg.seed = run_cfg.seed + run_cfg.rank
    run_cfg.save_eval_label = args.save_eval_label

    def mkdir():
        from datetime import datetime
        output_root = Path(args.output or cfg.project_root / 'output').absolute()
        run_cfg.output_dir = output_root / tags.dataset / tags.model / tags.experiment / tags.mode / tags.eval
        run_cfg.eval_dir = run_cfg.output_dir / 'eval'
        run_cfg.log_file = run_cfg.output_dir / f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

        run_cfg.ckpt_dir = output_root / tags.dataset / tags.model / tags.experiment / 'train/ckpt'
        run_cfg.ckpt_list = checkpoint.potential_ckpts(args.ckpt, default=run_cfg.ckpt_dir)
        assert run_cfg.ckpt_list
        run_cfg.ckpt_dir = Path(run_cfg.ckpt_list[-1]).parent

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
    logger.info(cfg.DATASET, title=[f'cfg.DATASET', 'VALUE'], width=50)
    logger.info(cfg.MODEL, title=[f'cfg.MODEL', 'VALUE'], width=50)
    logger.info(cfg.LR, title=[f'cfg.LR', 'VALUE'], width=50)
    logger.info(cfg.OPTIMIZATION, title=[f'cfg.OPTIMIZATION', 'VALUE'], width=50)
    logger.info(cfg.RUN, title=[f'cfg.MODEL', 'VALUE'], exclude=['workflows'], width=50)

    if cfg.RUN.tracker.get('init_kwargs', None):
        acc.init_trackers(cfg.RUN.tracker.project, config=cfg, init_kwargs=cfg.RUN.tracker.init_kwargs)

    """ build dataloaders & model """
    dataloader = build_dataloader(cfg.DATASET, cfg.RUN, training=False, logger=logger)
    model = build_detector(cfg.MODEL, dataset=dataloader.dataset)
    runner = DistRunner(cfg.RUN, model=model, logger=logger)
    logger.info(Hook.infos(), title=[f'hook', 'priority'])
    runner.run(ckpts=cfg.RUN.ckpt_list, dataloaders={cfg.RUN.mode: dataloader})

    logger.info("Done")
