import numpy as np
from pathlib import Path
from easydict import EasyDict

from rd3d.runner.ss3d.instance_bank import InstanceBank


def analysis(ins_bank: InstanceBank, full_annotated_dataset, threshold_list=(0.1, 0.5, 0.7)):
    from tqdm import tqdm

    def get_threshold_lists():
        ts = [0.0, *threshold_list, 1.0]
        ts.sort()
        return ts

    thresh = get_threshold_lists()
    class_names = full_annotated_dataset.class_names
    full_annotated_dataset.data_augmentor.data_augmentor_queue = []

    infos = {c: dict(gt=0, anno=0, pseudo=0, iou={k: 0 for k in thresh[:-1]})
             for c in class_names}

    for i, data_dict in enumerate(tqdm(iterable=full_annotated_dataset, desc='coverage ratio')):
        frame_id = data_dict['frame_id']

        gt_boxes = data_dict['gt_boxes'][:, :7]
        gt_labels = data_dict['gt_boxes'][:, 7]

        bank_boxes, anno_mask = ins_bank.get_scene(frame_id, return_points=False)
        pseudo_mask = np.logical_not(anno_mask)
        anno_labels = bank_boxes[anno_mask, 7]
        pseudo_boxes = bank_boxes[pseudo_mask, :7]
        pseudo_labels = bank_boxes[pseudo_mask, 7]

        iou = ins_bank.boxes_iou_cpu(pseudo_boxes, gt_boxes)

        for t in range(0, len(thresh) - 1):
            lower, higher = thresh[t], thresh[t + 1]
            iou_mask = np.logical_and(lower < iou, iou <= higher).any(axis=-1)
            for c, name in enumerate(class_names):
                label_mask = (c + 1) == pseudo_labels
                infos[name]['iou'][lower] += np.logical_and(iou_mask, label_mask).sum()

        for c, name in enumerate(class_names):
            infos[name]['gt'] += ((c + 1) == gt_labels).sum()
            infos[name]['anno'] += ((c + 1) == anno_labels).sum()
            infos[name]['pseudo'] += ((c + 1) == pseudo_labels).sum()

    thresh.pop(-1)
    return infos


def print_analysis(infos: dict):
    import prettytable

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    class_names = list(infos.keys())
    thresh = list(infos[class_names[0]]['iou'].keys())

    tb = prettytable.PrettyTable(
        title="instance bank information",
        field_names=['class', 'gt', 'anno', 'pseudo',
                     f'recall {[0] + thresh}', f'precision {[0] + thresh}']
    )
    tb.set_style(prettytable.SINGLE_BORDER)

    num_gt_all = sum([infos[c]['gt'] for c in class_names])
    num_anno_all = sum([infos[c]['anno'] for c in class_names])
    num_pseudo_all = sum([infos[c]['pseudo'] for c in class_names])
    num_match_all = np.zeros(len(thresh) + 1, dtype=int)
    for cls in class_names:
        num_gt, num_anno, num_pseudo = infos[cls]['gt'], infos[cls]['anno'], infos[cls]['pseudo']

        valid = np.array(list(infos[cls]['iou'].values()))
        valid = np.array([num_pseudo - valid.sum(), *valid])
        recall = valid / max(num_gt - num_anno, 1)
        precision = valid / max(num_pseudo, 1)

        tb.add_row((cls, num_gt, num_anno, num_pseudo, recall, precision))
        num_match_all += valid

    recall_all = num_match_all / max(num_gt_all - num_anno_all, 1)
    precision_all = num_match_all / max(num_pseudo_all, 1)
    tb.add_row(('all', num_gt_all, num_anno_all, num_pseudo_all, recall_all, precision_all))
    print(tb.get_string())


def instance_2d_mask_analysis(ins_bank: InstanceBank, dataset):
    import prettytable
    from collections import defaultdict
    from tqdm import tqdm

    mask_dir = ins_bank.root_path / 'training' / 'image_2_mask'
    if not mask_dir.exists():
        return

    infos = defaultdict(int)
    infos_dataset = defaultdict(int)
    for info in tqdm(iterable=dataset.kitti_infos, desc='mask 2d'):
        frame_id = info['point_cloud']['lidar_idx']
        anno = info['annos']

        for cls in anno['name']:
            infos_dataset[cls] += 1

        _, boxes_2d = dataset.get_pseudo_instances(frame_id)

        for box in boxes_2d:
            cls = dataset.class_names[int(box[-1]) - 1]
            infos[cls] += 1

    tb = prettytable.PrettyTable(
        title="2D instance masks information",
        field_names=list(infos.keys()) + ['all']
    )
    tb.set_style(prettytable.SINGLE_BORDER)
    tb.add_row(list(infos.values()) + [sum(infos.values())])

    tb = prettytable.PrettyTable(
        title="2D instance masks information",
        field_names=[k for k in infos] + ['all']
    )
    tb.set_style(prettytable.SINGLE_BORDER)
    tb.add_row(list(infos.values()) + [sum(infos.values())])
    print(tb.get_string())

    class_names = [c for c in infos_dataset]
    tb = prettytable.PrettyTable(
        title="dataset information",
        field_names=[k for k in class_names if k in infos] + ['other', 'all']
    )
    mask = np.array([k in infos for k in infos_dataset])
    v = np.array(list(infos_dataset.values()))
    tb.add_row(v[mask].tolist() + [v[np.logical_not(mask)].sum(), sum(infos_dataset.values())])
    tb.set_style(prettytable.SINGLE_BORDER)
    print(tb.get_string())


def main():
    import pickle
    from rd3d.datasets import build_dataloader
    from configs.base.datasets import kitti_3cls

    cache_file = "cache/bank_analysis.pkl"
    dataset = build_dataloader(kitti_3cls.DATASET, training=True)
    cfg = EasyDict(db_info_path='kitti_dbinfos_train.pkl',
                   bk_info_path='ss3d/bkinfos_train.pkl',
                   pseudo_database_path='ss3d/pseudo_database',
                   root_dir='data/kitti_sparse',
                   class_names=dataset.class_names)

    bank = InstanceBank(cfg)
    if Path(cache_file).exists():
        data = pickle.load(open(cache_file, 'rb'))
    else:
        data = analysis(bank, dataset)
        pickle.dump(data, open(cache_file, 'wb'))

    print_analysis(data)
    instance_2d_mask_analysis(bank, dataset)


if __name__ == '__main__':
    main()
