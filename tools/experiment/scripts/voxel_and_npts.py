import numpy as np
import torch
import pickle
import argparse
from pathlib import Path


def parse_args():
    from rd3d.api import Config, set_random_seed

    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, help='specify the config for training')
        parser.add_argument('--seed', type=int, default=None,
                            help='random seed')
        parser.add_argument('--iter', type=int, default=0)
        return parser

    def parse_config(args):
        """read config from file and cmdline"""
        import time
        cfg = Config.fromfile(args.cfg)
        cfg.RUN.seed = args.seed if args.seed is not None else time.time_ns() % (2 ** 32 - 1)
        cfg.sample_iter = args.iter
        return cfg

    args = add_args().parse_args()
    cfg = parse_config(args)
    set_random_seed(cfg.RUN.seed)
    return args, cfg


def build_sampler(**kwargs):
    from rd3d.models.backbones_3d.pfe.ops.point_sampler import sampler
    from easydict import EasyDict
    return sampler.from_cfg(EasyDict(name='hvcs_v2_info', **kwargs))


def exp(dataloader):
    from tqdm import tqdm
    from rd3d.utils.common_utils import gather
    voxel = [2.0, 2.0, 2.0]
    sampler1 = build_sampler(sample=16384, voxel=voxel, max_iter=1000, tolerance=0.001)
    sampler2 = build_sampler(sample=4096, voxel=voxel, max_iter=1000, tolerance=0.001)
    voxels1 = []
    voxels2 = []
    sizes = []
    for batch_dict in tqdm(iterable=dataloader):
        dataloader.dataset.load_data_to_gpu(batch_dict)
        bs, cnl = batch_dict['batch_size'], batch_dict['points'].shape[-1]
        xyz1 = batch_dict['points'].view(bs, -1, cnl)[..., 1:4].contiguous()

        ind1, voxel1, _ = sampler1(xyz1)
        xyz2 = gather(xyz1, ind1)
        ind2, voxel2, _ = sampler2(xyz2)
        voxels1.extend(voxel1.cpu().numpy().tolist())
        voxels2.extend(voxel2.cpu().numpy().tolist())

        size = batch_dict['gt_boxes'][..., 3:6]
        size = size[size.sum(dim=-1) != 0]
        sizes.extend(size.cpu().numpy().tolist())

    print("voxel 65535 -> 16384: {}".format(np.array(voxels1).mean(axis=0)))
    print("voxel 16384 -> 4096: {}".format(np.array(voxels2).mean(axis=0)))
    print("size: {}".format(np.array(sizes).mean(axis=0)))


def interpolate(raw_x, raw_y, out_res):
    from scipy import interpolate
    us_coord = np.linspace(raw_x.min(), raw_x.max(), out_res).reshape(-1)
    return us_coord, interpolate.interp1d(raw_x.reshape(-1), raw_y.reshape(-1), kind="quadratic")(us_coord)


def show_figures(infos):
    from matplotlib import pyplot as plt
    mean = []
    std = []
    for k in infos:
        v = np.array(infos[k])[:, 0]
        mean.append(v.mean())
        std.append(v.std())
    mean = np.array(mean)
    std = np.array(std)
    scale = 1.0
    plt.figure(figsize=(6.4 * 0.7 * scale, 4.8 * 0.20 * scale))
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    x = np.array(range(1, len(mean) + 1))
    plt.fill_between(x, mean + std, mean - std, color='b', alpha=0.2)
    plt.plot(x, mean, color='b', alpha=0.8)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xlabel('the number of non-empty voxels', labelpad=0)
    plt.ylabel(r'$v_x$ (m)')
    # plt.xticks([10 * 1, 10 ** 3], rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(left=0.14 / scale, bottom=0.445 / scale, top=0.98)
    plt.savefig('tools/experiments/results/voxel_and_npts.pdf')
    plt.show()


def main():
    """ experiments setup """
    cfg.RUN.samples_per_gpu = 8
    cfg.result_path = "tools/experiments/data/voxel_npts.pkl"

    """ run exp or read from cache """
    if Path(cfg.result_path).exists():
        voxels = pickle.load(open(cfg.result_path, 'rb'))
    else:
        """ build dataset"""
        from rd3d import build_dataloader
        dataloader = build_dataloader(cfg.DATASET, cfg.RUN, training=True, logger=logger)
        voxels = exp(dataloader)
        pickle.dump(voxels, open(cfg.result_path, 'wb'))
    show_figures(voxels)


if __name__ == '__main__':
    from rd3d.api import create_logger

    args, cfg = parse_args()
    logger = create_logger()
    logger.info(f"seed: {cfg.RUN.seed}")

    main()
