import open3d
import argparse
import numpy as np
from easydict import EasyDict
from rd3d.utils import viz_utils
from rd3d.datasets import build_dataloader
from configs.base.datasets import kitti_3cls
from rd3d.runner.ss3d.instance_bank import InstanceBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_id", nargs="+", type=str)
    used_frame_id = parser.parse_args().frame_id

    dataset = build_dataloader(kitti_3cls.DATASET, training=True)
    dataset.data_augmentor.data_augmentor_queue = []
    frame_id_to_index = {info['point_cloud']['lidar_idx']: i for i, info in enumerate(dataset.kitti_infos)}

    cfg = EasyDict(db_info_path='kitti_dbinfos_train.pkl',
                   bk_info_path='ss3d/bkinfos_train.pkl',
                   pseudo_database_path='ss3d/pseudo_database',
                   root_dir='data/kitti_sparse',
                   class_names=['Car', 'Pedestrian', 'Cyclist'])
    bank = InstanceBank(cfg)

    for frame_id in used_frame_id or frame_id_to_index.keys():
        data_dict = dataset[frame_id_to_index[frame_id]]

        obj_pts, ins_boxes, anno_mask = bank.get_scene(frame_id, return_points=True)
        ins_boxes = ins_boxes[:, :7]
        colors = np.ones_like(ins_boxes[:, :3]) * np.array([1, 0, 0])
        colors[anno_mask] = np.array([0, 0, 1])

        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'][:, :7]
        gt_boxes = gt_boxes[np.logical_not(np.isclose(gt_boxes[:, 0], ins_boxes[anno_mask, 0]))]
        gt_colors = np.ones_like(gt_boxes[:, :3]) * np.array([0, 1, 0])

        boxes = np.concatenate((gt_boxes, ins_boxes,), axis=0)
        colors = np.concatenate((gt_colors, colors,), axis=0)
        # key_points = np.vstack(obj_pts)
        key_points = None
        viz_utils.viz_scenes((points, (boxes, colors), key_points), title="bank visualization")

        if dataset.root_split_path / 'image_2d_mask':
            import cv2 as cv
            from matplotlib import pyplot as plt
            from rd3d.utils import box_utils
            img = dataset.get_image(frame_id)
            masks, labels, calib = dataset.get_pseudo_instances(frame_id, return_calib=True)
            factor = 255 / max(1, masks.max())
            masks = (masks * factor).astype(np.uint8)
            masks = cv.cvtColor(masks, cv.COLOR_GRAY2BGR)

            corners2d = box_utils.boxes2d_to_corners_2d(labels)
            corners2d_lines = box_utils.corners2d_to_lines(corners2d)
            for k, lines in enumerate(corners2d_lines.astype(int)):
                for line in lines:
                    cv.line(masks, line[0].tolist(), line[1], (0, 255, 0), 1)
                cv.putText(masks, f"{k}", lines[2][0].tolist(),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            h, w = masks.shape[:2]
            plt.figure(figsize=(int(w / 100), 2 * int(h / 100)))
            plt.axis('off')
            plt.subplot(2, 1, 1)
            plt.imshow(masks)
            plt.subplot(2, 1, 2)
            plt.imshow(img)
            plt.tight_layout()
            plt.show()
            plt.close()


if __name__ == '__main__':
    main()
