# !/usr/bin/env python3
from utils import load_plugins

import time
import torch
import numpy as np
import tensorrt as trt

import rospy
import ros_numpy
from sensor_msgs.point_cloud2 import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from rospkg.rospack import get_package_name

import threading

model_path = "tools/models/trt/iassd_hvcsx2_gqx2_4x2_80e_peds_sim.engine"
ros_topic_in = "/sensor_scan"
ros_topic_out = "objects"

args = {'header': None}


def loop():
    import pycuda.driver as cuda
    import pycuda.autoinit

    class Interface:
        def __init__(self):
            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
            with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.ctx.set_optimization_profile_async(0, self.stream.handle)
            self.input_shape = self.ctx.get_binding_shape(self.engine.get_binding_index("points"))
            self.input_shape[0] = 1
            self.ctx.set_binding_shape(self.engine.get_binding_index("points"), self.input_shape)
            self.pc = np.zeros([i for i in self.input_shape], dtype=np.float32)
            self.points = np.random.uniform(-50, 50, [i for i in self.input_shape])[0]
            self.h_inputs = {'points': self.pc}
            self.d_inputs = {}
            self.h_outputs = {}
            self.d_outputs = {}
            self.shape_outputs = {}
            for binding in self.engine:
                if self.engine.binding_is_input(binding):
                    self.d_inputs[binding] = cuda.mem_alloc(self.h_inputs[binding].nbytes)
                else:
                    output_shape = self.ctx.get_binding_shape(self.engine.get_binding_index(binding))
                    self.shape_outputs[binding] = [i for i in output_shape]
                    size = trt.volume(output_shape)
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    self.h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                    self.d_outputs[binding] = cuda.mem_alloc(self.h_outputs[binding].nbytes)
            assert self.ctx.all_binding_shapes_specified
            self.inference()  # wamrup
            self.ready = False

        def set_points(self, points):
            self.points = points
            self.ready = True

        def inference(self):
            self.pc[0, :self.points.shape[0], :3] = self.points[:, :3]
            self.h_inputs = {'points': self.pc}
            for key in self.h_inputs:
                cuda.memcpy_htod_async(self.d_inputs[key], self.h_inputs[key], self.stream)

            self.ctx.execute_async_v2(
                bindings=[int(self.d_inputs[k]) for k in self.d_inputs] + [int(self.d_outputs[k]) for k in
                                                                           self.d_outputs],
                stream_handle=self.stream.handle
            )
            for key in self.h_outputs:
                cuda.memcpy_dtoh_async(self.h_outputs[key], self.d_outputs[key], self.stream)
            self.stream.synchronize()

            boxes = self.h_outputs['boxes'].reshape(*self.shape_outputs['boxes'])[0]
            num = self.h_outputs['nums'].reshape(-1)[0]
            print(boxes.shape,num)
            center = np.asarray(boxes[:num, :3], dtype=np.float32)
            boxes = np.asarray(boxes[:num, :7], dtype=np.float32)
            self.center = center
            self.boxes = boxes
            self.ready = False

    class DNNDetector:
        def __init__(self):
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
                corner_for_box_list = [0, 1, 0, 3, 2, 3, 2, 1, 4, 5, 4, 7, 6, 7, 6, 5, 3, 7, 0, 4, 1, 5, 2, 6, 0, 5, 1,
                                       4]
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
            args['header'] = points_msg.header
            points_np_struct = ros_numpy.numpify(points_msg)
            points_np = np.zeros((points_np_struct.size, 4), dtype=np.float32)
            points_np[:, 0] = points_np_struct['x'].ravel()
            points_np[:, 1] = points_np_struct['y'].ravel()
            points_np[:, 2] = points_np_struct['z'].ravel()
            points_np[:, 3] = 0
            print("points numbers : ", points_np_struct.size)
            try:
                points_np[:, 3] = points_np_struct['intensity']
            except:
                pass
            if points_np.shape[0] < 100:
                return
            self.model.set_points(points_np)

    dnn_detector = DNNDetector()
    while not rospy.is_shutdown():
        time.sleep(0.001)
        if dnn_detector.model.ready:
            t1 = time.time()
            dnn_detector.model.inference()
            t2 = time.time()
            boxes = dnn_detector.model.boxes
            print(f"runtime: {(t2 - t1) * 1000} ms, boxes: {boxes.shape[0]}")
            markers = dnn_detector.boxes2markers(args['header'], boxes)
            dnn_detector.pub.publish(markers)


if __name__ == "__main__":
    rospy.init_node('object_detection')
    thread = threading.Thread(target=loop)
    thread.start()
    rospy.spin()
    thread.join()
