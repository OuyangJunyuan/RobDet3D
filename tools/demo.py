import time
import argparse
import numpy as np

from rd3d.utils.base import Hook
from rd3d.api import set_random_seed, Config, checkpoint, create_logger
from rd3d import build_detector, build_dataloader
from rd3d.utils.viz_utils import viz_scenes

import torch


@Hook.auto_call
def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs='...', help='set extra config keys if needed')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--scenes', type=int, default=2, help='random seed')
    return parser


@Hook.auto_call
def parse_config(args):
    """read config from file and cmdline"""
    cfg = Config.fromfile(args.cfg_file)
    cfg = Config.merge_custom_cmdline_setting(cfg, args.set_cfgs) if args.set_cfgs is not None else cfg
    cfg.RUN.seed = args.seed if args.seed is not None else time.time_ns() % (2 ** 32 - 1)
    return cfg


if __name__ == '__main__':
    """ init config """
    args = add_args().parse_args()
    cfg = parse_config(args)
    set_random_seed(cfg.RUN.seed)
    logger = create_logger(stderr=True)
    logger.info(f"seed: {cfg.RUN.seed}")
    """ build dataloaders & model & optimizer & lr_scheduler """
    dataset = build_dataloader(cfg.DATASET, training=False, logger=logger)
    model = build_detector(cfg.MODEL, dataset=dataset)
    checkpoint.load_from_file(args.ckpt, model)
    model.cuda()
    model.eval()
    scenes = np.random.randint(0, len(dataset), args.scenes)  # [200]  #
    with torch.no_grad():
        for ind in scenes:
            batch_dict = dataset.collate_batch([dataset[ind]])
            dataset.load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            logger.info(f"scenes: {batch_dict['frame_id']}")
            viz_scenes((batch_dict['points'].view(-1, 5), batch_dict['gt_boxes'].view(-1, 8)),
                       (batch_dict['points'].view(-1, 5), pred_dicts[0]['pred_boxes'].view(-1, 7)),
                       offset=(0, 35, 0))
    logger.info("Done")
