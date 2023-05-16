import argparse
import time

import numpy as np
from easydict import EasyDict
from rd3d.utils import viz_utils
from rd3d.datasets import build_dataloader
from configs.base.datasets import kitti_3cls
from rd3d.runner.ss3d.instance_bank import InstanceBank
import threading
import cv2 as cv
from matplotlib import pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class BankVisualizer:
    def __init__(self):
        self.window = gui.Application.instance.create_window(
            "Bank Visualization", 1920, 1080)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.5, 0.5, 0.5, 0.5])
        self.widget3d.scene.show_axes(True)
        self.widget3d.scene.scene.enable_sun_light(False)
        # self.widget3d.scene.view.set_post_processing(False)
        self.widget3d.set_on_key(self._on_key)
        self.widget3d.setup_camera(
            60, o3d.geometry.AxisAlignedBoundingBox([-0, -20, 0], [20, 20, 20]), [0, 0, 0]
        )
        self.window.add_child(self.widget3d)

        random_img = o3d.geometry.Image(np.clip(np.random.random([100, 100, 3]) * 255, 0, 255).astype(np.uint8))

        em = self.window.theme.font_size
        margin = 2.5 * em
        self.panel = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.panel.add_child(gui.Label("image"))
        self.img = random_img
        self.image_widget = gui.ImageWidget(self.img)
        self.panel.add_child(self.image_widget)
        self.show_masks = (dataset.root_split_path / 'image_2_mask').exists()

        if self.show_masks:
            self.panel.add_child(gui.Label("mask"))
            self.mask = random_img
            self.mask_widget = gui.ImageWidget(self.img)
            self.panel.add_child(self.mask_widget)
            self.window.add_child(self.panel)

        self.labels = []
        self.is_done = False
        self.is_update = False
        self.frame_idx = 0
        threading.Thread(target=self._update_thread).start()

    def _on_layout(self, layout_context):
        # content_rect = self.window.content_rect
        # panel_width = 40 * layout_context.theme.font_size  # 15 ems wide
        # self.widget3d.frame = gui.Rect(content_rect.x, content_rect.y,
        #                                content_rect.width - panel_width,
        #                                content_rect.height)
        # self.panel.frame = gui.Rect(self.widget3d.frame.get_right(),
        #                             content_rect.y, panel_width,
        #                             content_rect.height)
        content_rect = self.window.content_rect
        panel_high = 15 * layout_context.theme.font_size  # 15 ems wide
        self.widget3d.frame = gui.Rect(content_rect.x, content_rect.y,
                                       content_rect.width,
                                       content_rect.height - panel_high)
        self.panel.frame = gui.Rect(content_rect.x, self.widget3d.frame.get_bottom(),
                                    content_rect.width, panel_high)

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_key(self, e):
        if e.key == gui.KeyName.RIGHT and e.type == gui.KeyEvent.UP:
            self.is_update = False
            self.frame_idx += 1
            print("->")
            return gui.Widget.EventCallbackResult.HANDLED
        if e.key == gui.KeyName.LEFT and e.type == gui.KeyEvent.UP:
            self.is_update = False
            self.frame_idx -= 1
            print("<-")
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _update_thread(self):
        color_map = {'gt': [0, 1.0, 0], 'anno': [0, 0, 1.0], 'pseudo': [1.0, 0, 0]}

        while True:
            while self.is_update:
                time.sleep(0.1)
                if self.is_done:
                    return
            self.frame_idx = np.clip(self.frame_idx, 0, len(frame_ids) - 1)
            frame_id = frame_ids[self.frame_idx]
            data_dict = dataset[frame_id_to_index[frame_id]]

            points = data_dict['points']
            gt_boxes = data_dict['gt_boxes']

            ins_boxes, anno_mask = bank.get_scene(frame_id, return_points=False)
            gt_boxes = gt_boxes[np.logical_not(np.isclose(gt_boxes[:, 0], ins_boxes[anno_mask, 0]))]
            gt_labels = gt_boxes[:, 7]
            gt_boxes = gt_boxes[:, :7]
            gt_colors = np.ones_like(gt_boxes[:, :3]) * np.array(color_map['gt'])

            ins_labels = ins_boxes[:, 7]
            ins_boxes = ins_boxes[:, :7]
            colors = np.ones_like(ins_boxes[:, :3]) * np.array(color_map['pseudo'])
            colors[anno_mask] = np.array(color_map['anno'])
            boxes = np.concatenate((gt_boxes, ins_boxes,), axis=0)
            colors = np.concatenate((gt_colors, colors,), axis=0)
            labels = np.concatenate((gt_labels, ins_labels), axis=0)

            import cv2 as cv
            from rd3d.utils import box_utils
            from rd3d.datasets.kitti.kitti_utils import calib_to_matricies

            def homo(x):
                return np.concatenate((x, np.ones_like(x[..., :1])), axis=-1)

            def lidar_to_img(x, num_corners=4):
                x = homo(x) @ mat_lidar2cam.T
                x = x @ mat_cam2img.T
                x = x[..., :2] / x[..., 2:3]
                if num_corners == 8:
                    return x
                elif num_corners == 4:
                    return box_utils.corners3d_to_corners2d(x)
                else:
                    raise NotImplementedError

            img = (dataset.get_image(frame_id) * 255 / 2).astype(np.uint8)
            mat_lidar2cam, mat_cam2img = calib_to_matricies(dataset.get_calib(frame_id))
            corners3d = box_utils.boxes3d_to_corners_3d(boxes)
            corners3d_img = lidar_to_img(corners3d, num_corners=8)
            corners3d_img_lines = box_utils.corners3d_to_lines(corners3d_img)
            for k, lines in enumerate(corners3d_img_lines.astype(int)):
                color = colors[k]
                for line in lines:
                    cv.line(img, line[0].tolist(), line[1], color * 255, 2)

            self.img = o3d.geometry.Image(img)

            if self.show_masks:
                masks, boxes2d = dataset.get_pseudo_instances(frame_id)
                factor = 255 / max(1, masks.max())
                masks = (masks * factor).astype(np.uint8)
                masks = cv.cvtColor(masks, cv.COLOR_GRAY2RGB)

                corners2d = box_utils.boxes2d_to_corners_2d(boxes2d)
                corners2d_lines = box_utils.corners2d_to_lines(corners2d)
                for k, lines in enumerate(corners2d_lines.astype(int)):
                    for line in lines:
                        cv.line(masks, line[0].tolist(), line[1], [255, 255, 255], 1)
                    cv.putText(masks, f"{k}", lines[2][0].tolist(), cv.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255])

                corners3d = box_utils.boxes3d_to_corners_3d(boxes)
                corners3d_img = lidar_to_img(corners3d, num_corners=4)
                corners2d_img_lines = box_utils.corners2d_to_lines(corners3d_img)
                for k, lines in enumerate(corners2d_img_lines.astype(int)):
                    color = colors[k]
                    for line in lines:
                        cv.line(masks, line[0].tolist(), line[1], color * 255, 1)
                self.mask = o3d.geometry.Image(masks)

            self.is_update = True

            def update():
                self.window.title = frame_id
                self.widget3d.scene.clear_geometry()
                for text in self.labels:
                    self.widget3d.remove_3d_label(text)
                point_mat = rendering.MaterialRecord()
                point_mat.point_size = 2
                viz_utils.add_points(vis=self.widget3d.scene, points=points, name="points", material=point_mat)
                boxes_mat = rendering.MaterialRecord()
                boxes_mat.shader = "unlitLine"  # to enable line_width
                boxes_mat.line_width = 5
                viz_utils.add_boxes(vis=self.widget3d.scene, boxes=boxes, color=colors, name="boxes",
                                    material=boxes_mat)

                for i in range(len(labels)):
                    pos = [boxes[i, 0], boxes[i, 1], boxes[i, 2] + 5]
                    text = self.widget3d.add_3d_label(pos, dataset.class_names[int(labels[i]) - 1])
                    text.color = gui.Color(*colors[i])
                    self.labels.append(text)

                self.image_widget.update_image(self.img)
                if self.show_masks:
                    self.mask_widget.update_image(self.mask)

                bounds = self.widget3d.scene.bounding_box
                center = np.asarray(bounds.get_center())
                look_at = center
                eye = [-20, center[1], 20]
                self.widget3d.look_at(look_at, eye, [1, 0, 0])

            gui.Application.instance.post_to_main_thread(self.window, update)


if __name__ == '__main__':
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

    frame_ids = used_frame_id or list(frame_id_to_index.keys())

    app = gui.Application.instance
    app.initialize()

    win = BankVisualizer()
    app.run()
