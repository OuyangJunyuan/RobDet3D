# !/usr/bin/env python3
import time

import torch
from rd3d import PROJECT_ROOT
from rd3d.datasets import DatasetTemplate
from rd3d.api import Config, checkpoint, create_logger
from rd3d import build_detector

import yaml
import numpy as np
from easydict import EasyDict
from importlib import import_module

import rospy
import ros_numpy
from sensor_msgs.point_cloud2 import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from rospkg.rospack import get_package_name

cfg_file = PROJECT_ROOT / 'configs/iassd/wzh/iassd_hvcsx2_gqx2_exp.py'
ckpt_file = PROJECT_ROOT / 'tools/models/wzh/iassd_hvcsx2_gqx2_exp_1x4_80e_kitti_peds_fov90(default)(kitti-pretrain).pth'
# ros_topic_in = "/ouster/points"
ros_topic_in = "/os1_cloud_node/points"
ros_topic_out = "objects"


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return {}


class Interface:
    def __init__(self):
        cfg = Config.fromfile(cfg_file)

        self.logger = create_logger(name="ros")
        self.logger.disabled = True

        self.demo_dataset = DemoDataset(dataset_cfg=cfg.DATASET, class_names=cfg.DATASET.CLASS_NAMES, training=False)
        self.model = build_detector(cfg.MODEL, dataset=self.demo_dataset)
        checkpoint.load_from_file(ckpt_file, self.model)
        self.model.cuda()
        self.model.eval()

    def inference(self, points):
        with torch.no_grad():
            # prepare data and load data to gpu
            print("======")
            input_dict = {'points': points}
            data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
            data_dict = self.demo_dataset.collate_batch([data_dict])
            self.demo_dataset.load_data_to_gpu(data_dict)

            # inference once in batch size : 1
            pred_dicts, _ = self.model.forward(data_dict)
            pred_dicts = pred_dicts[0]

            # analysis the result
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            pred_iou = pred_boxes[..., -1]
            pred_boxes = pred_boxes[..., :7]
            intersection = 2 - 2 / (1 + pred_iou)  # iou -> intersection
            pred_bound = (1.5 - intersection) * np.linalg.norm(pred_boxes[..., 3:5], axis=-1)

            # print(f"boxes: {pred_boxes}")
            print(f"labels: {pred_labels}")
            print(f"scores: {pred_scores}")
            print(f"boundaries: {pred_bound}")
            print("======")
            return pred_boxes, pred_labels, pred_scores, pred_bound


class DNNDetector:
    def __init__(self):
        rospy.init_node('object_detection')
        self.model = Interface()
        self.sub = rospy.Subscriber(ros_topic_in, PointCloud2, self.handler, queue_size=1)
        self.pub = rospy.Publisher(ros_topic_out, MarkerArray, queue_size=1)

    def boxes2markers(self, header, boxes):
        def get_default_marker(action, ns):
            marker = Marker()
            marker.header = header
            marker.type = marker.LINE_LIST
            marker.action = action

            marker.ns = ns
            # marker scale (scale y and z not used due to being linelist)
            marker.scale.x = 0.08
            # marker color
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.points = []
            return marker

        def get_boxes_corners_line_list():
            from rd3d.utils.box_utils import boxes_to_corners_3d
            corners3d = boxes_to_corners_3d(boxes[:, :7])  # (N,8,3)
            corner_for_box_list = [0, 1, 0, 3, 2, 3, 2, 1, 4, 5, 4, 7, 6, 7, 6, 5, 3, 7, 0, 4, 1, 5, 2, 6, 0, 5, 1, 4]
            boxes_corners_line_list = corners3d[:, corner_for_box_list, :3]
            return boxes_corners_line_list

        markers = MarkerArray()
        corners = get_boxes_corners_line_list()
        for i, box in enumerate(boxes):
            marker = get_default_marker(Marker.ADD, 'Pedestrian')
            marker.id = i
            markers.markers.append(marker)
            for box_corner in corners[i]:
                marker.points.append(Point(box_corner[0], box_corner[1], box_corner[2]))

        if not hasattr(self, "last_markers"):
            self.last_markers = []
        if len(markers.markers) < len(self.last_markers):
            for last_marker in self.last_markers[len(markers.markers):]:
                last_marker.action = Marker.DELETE
                markers.markers.append(last_marker)
        self.last_markers = markers.markers
        return markers

    def handler(self, points_msg):
        points_np_struct = ros_numpy.numpify(points_msg)
        points_np = np.zeros((points_np_struct.size, 4), dtype=np.float32)
        points_np[:, 0] = points_np_struct['x'].ravel()
        points_np[:, 1] = points_np_struct['y'].ravel()
        points_np[:, 2] = points_np_struct['z'].ravel()
        points_np[:, 3] = 0
        # try:
        #     points_np[:, 3] = points_np_struct['intensity']
        # except:
        #     pass

        if points_np.shape[0] < 100:
            return
        t1 = time.time()
        boxes, labels, scores, boundaries = self.model.inference(points_np)  # numpy in & numpy out
        print((time.time() - t1) * 1000)
        markers = self.boxes2markers(points_msg.header, boxes)
        self.pub.publish(markers)


if __name__ == "__main__":
    dnn_detector = DNNDetector()
    # points = np.fromfile(PROJECT_ROOT / 'tools/demo_data/000000.bin', dtype=np.float32)
    # dnn_detector.model.inference(points.reshape(-1, 4))
    rospy.spin()
