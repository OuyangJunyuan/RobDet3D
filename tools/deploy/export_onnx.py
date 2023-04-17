import torch
from utils import load_plugins


def demo():
    import time
    import argparse
    from pathlib import Path
    from rd3d.utils.base import Hook
    from rd3d.api import checkpoint, Config, create_logger, set_random_seed
    from rd3d import build_detector, build_dataloader

    @Hook.auto
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg_file', type=Path, required=True, help='specify the config for training')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        parser.add_argument('--onnx', type=Path, required=True, help='the path of output')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs='...', help='set extra config keys if needed')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        return parser

    @Hook.auto
    def parse_config(args):
        """read config from file and cmdline"""
        cfg = Config.fromfile(args.cfg_file)
        cfg = Config.merge_custom_cmdline_setting(cfg, args.set_cfgs) if args.set_cfgs is not None else cfg
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
    checkpoint.load_from_file(args.ckpt, model)
    return model, dataloader, args


def export_onnx():
    data_dict = dataloader.dataset[0]
    batch_dict = dataloader.dataset.collate_batch([data_dict])
    dataloader.dataset.load_data_to_gpu(batch_dict)
    batch_dict = dict(points=batch_dict['points'].view(1, -1, 5)[..., 1:].contiguous())
    output_path = fold / (file + '.onnx')
    with torch.no_grad():
        torch.onnx.export(model,
                          (batch_dict, {}),
                          output_path,
                          input_names=['points'],
                          output_names=['boxes', 'scores', 'nums'],
                          dynamic_axes={'points': {0: 'batch_size'},
                                        'boxes': {0: 'batch_size'},
                                        'scores': {0: 'batch_size'},
                                        'nums': {0: 'batch_size'}}
                          )
    return output_path


if __name__ == "__main__":
    model, dataloader, args = demo()
    model.cuda()
    model.eval()

    if args.onnx.suffix:
        fold = args.onnx.parent
        file = args.onnx.stem
    else:
        fold = args.onnx
        file = args.cfg_file.stem
    fold.mkdir(parents=True, exist_ok=True)

    save_path = export_onnx()
    print(f"save onnx model: {save_path}")
