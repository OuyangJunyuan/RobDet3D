import time
import argparse
from tqdm import tqdm
from rd3d.utils.base import Hook
from rd3d.api import set_random_seed, Config, checkpoint, create_logger
from rd3d import build_detector, build_dataloader

import torch


@Hook.auto
def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--batch', type=int, default=None, help='random seed')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    return parser


@Hook.auto
def parse_config(args):
    """read config from file and cmdline"""
    cfg = Config.fromfile(args.cfg_file)
    cfg.RUN.seed = args.seed if args.seed is not None else time.time_ns() % (2 ** 32 - 1)
    cfg.RUN.samples_per_gpu = args.batch if args.batch is not None else cfg.RUN.samples_per_gpu
    return cfg


if __name__ == '__main__':
    """ init config """
    args = add_args().parse_args()
    cfg = parse_config(args)
    set_random_seed(cfg.RUN.seed)
    logger = create_logger()
    logger.info(f"seed: {cfg.RUN.seed}")
    """ build dataloaders & model & optimizer & lr_scheduler """
    dataloader = build_dataloader(cfg.DATASET, run_cfg=cfg.RUN, training=False, logger=logger)
    model = build_detector(cfg.MODEL, dataset=dataloader.dataset)
    checkpoint.load_from_file(args.ckpt, model)
    model.cuda()
    model.eval()
    warmup = len(dataloader) // 2
    with torch.no_grad():
        start_time = time.time()
        warmup_num = 0
        for batch_dict in tqdm(iterable=dataloader, desc='runtime'):
            if warmup_num <= warmup: start_time = time.time()
            warmup_num += 1
            dataloader.dataset.load_data_to_gpu(batch_dict)
            model(batch_dict)
        end_time = time.time()
        runtime = (end_time - start_time) * 1000.0 / ((len(dataloader) - warmup) * dataloader.batch_size)
        logger.info('inference time: %.2fms' % runtime)
    logger.info("Done")
