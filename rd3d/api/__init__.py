from .config import Config
from .dist import acc, get_dist_state, on_rank0
from .log import create_logger

__all__ = ['Config',
           'acc', 'get_dist_state', 'on_rank0',
           'create_logger', 'set_random_seed']


def set_random_seed(seed=0, deterministic=False):
    import random
    import numpy
    import torch
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def demo(parser=None):
    import time
    import argparse
    from ..utils.base import Hook
    from . import checkpoint
    from rd3d import build_detector, build_dataloader

    @Hook.auto
    def add_args(parser=parser):
        parser = argparse.ArgumentParser() if parser is None else parser
        parser.add_argument('--cfg_file', type=str, default="configs/iassd/iassd_hvcsx1_4x8_80e_kitti_3cls.py",
                            help='specify the config for training')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs='...', help='set extra config keys if needed')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        parser.add_argument('--batch', type=int, default=None, help='batch_size')
        return parser

    @Hook.auto
    def parse_config(args):
        """read config from file and cmdline"""
        cfg = Config.fromfile(args.cfg_file)
        cfg = Config.merge_custom_cmdline_setting(cfg, args.set_cfgs) if args.set_cfgs is not None else cfg
        cfg.RUN.samples_per_gpu = args.batch if args.batch is not None else cfg.RUN.samples_per_gpu
        cfg.RUN.seed = args.seed if args.seed is not None else time.time_ns() % (2 ** 32 - 1)
        return cfg

    args = add_args().parse_args()
    cfg = parse_config(args)
    set_random_seed(cfg.RUN.seed)
    logger = create_logger(stderr=False)
    logger.info(f"seed: {cfg.RUN.seed}")
    """ build dataloaders & model """
    dataloader = build_dataloader(cfg.DATASET, cfg.RUN, training=False, logger=logger)
    model = build_detector(cfg.MODEL, dataset=dataloader.dataset)
    if args.ckpt:
        checkpoint.load_from_file(args.ckpt, model)
    if parser:
        return model, dataloader, args
    else:
        return model, dataloader
