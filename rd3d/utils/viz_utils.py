import open3d
import numpy as np


def add_points(vis, points, colors=None, offset=None):
    def color_map(x):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('turbo')
        x = x - x.min()
        # x /= x.max()
        x = cmap(x)[:, :3].reshape(-1, 3)  # rgba -> rgb
        return x

    xyz = points[:, :3]
    if offset is not None:
        xyz += offset

    if colors is None:
        if points.shape[-1] > 3:
            colors = color_map(points[:, -1])
        else:
            colors = open3d.utility.Vector3dVector()
    elif isinstance(colors, list):
        colors = np.array(colors).reshape(1, 3)
        colors = np.ones_like(xyz) * colors

    o3d_geometry = open3d.geometry.PointCloud()
    o3d_geometry.points = open3d.utility.Vector3dVector(np.ascontiguousarray(xyz))
    o3d_geometry.colors = open3d.utility.Vector3dVector(np.clip(np.ascontiguousarray(colors), 0, 1))
    vis.add_geometry(o3d_geometry)


def add_keypoint(vis, points, radius=0.05, color=None, offset=None):
    for i in range(points.shape[0]):
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        xyz = points[i, :3]
        if offset is not None:
            xyz += offset
        mesh_sphere.translate(xyz)
        mesh_sphere.compute_vertex_normals()
        if color is None:
            mesh_sphere.paint_uniform_color([0.9, 0.6, 0.2])
        else:
            mesh_sphere.paint_uniform_color(np.clip(color, 0, 1))
        vis.add_geometry(mesh_sphere)


def add_boxes(vis, boxes, labels=None, scores=None, color=None, offset=None):
    def translate_boxes_to_open3d_instance(gt_boxes):
        """
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0
        """
        center = gt_boxes[0:3]
        if offset is not None:
            center += offset
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
        box3d.color = (0.5, 0.5, 0.5)  # remove Open3D warning
        line_set = open3d.geometry.LineSet()
        line_set = line_set.create_from_oriented_bounding_box(box3d)

        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = open3d.utility.Vector2iVector(lines)
        return line_set

    box_colormap = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
    ]
    boxes, fets = boxes[:, :7], boxes[:, 7:]
    if labels is None and fets.shape[1] > 0: labels = fets[:, 0].astype(int)
    if scores is None and fets.shape[1] > 1: scores = fets[:, 1]

    for i in range(boxes.shape[0]):
        box_lines = translate_boxes_to_open3d_instance(boxes[i])
        if color is None:
            c = box_colormap[labels[i]] if labels is not None else (1, 0, 0)
        else:
            c = color[i] if len(color.shape) and i < color.shape[0] else color
        c = np.array(np.clip(c, 0, 1))

        num_lines = np.asarray(box_lines.lines).shape[0]

        box_lines.colors = open3d.utility.Vector3dVector(np.repeat(c[None, ...], repeats=num_lines, axis=0))
        vis.add_geometry(box_lines)
    return vis


def add_scene(origin=True, title='Open3D'):
    vis = open3d.visualization.Visualizer()
    vis.create_window(title)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3) * 0.5
    if origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    return vis


def viz_points(points):
    vis = add_scene()

    if points.ndim == 3:
        add_points(vis, points[-1, :, :3].cpu().numpy(), None)
    else:
        add_points(vis, points[:, :3].cpu().numpy(), None)
    vis.run()
    vis.destroy_window()


def viz_scene(points=None, boxes=None, key_points=None, center=None, vis=None):
    def check(obj):
        to_numpy = lambda x: x.copy() if isinstance(x, np.ndarray) else x.cpu().numpy().copy()
        if obj is None:
            return None, None
        if isinstance(obj, (list, tuple)):
            obj, obj_color = obj
            return to_numpy(obj), obj_color
        else:
            return to_numpy(obj), None

    points, points_color = check(points)
    boxes, boxes_color = check(boxes)
    key_points, key_points_color = check(key_points)

    show = vis is None
    vis = add_scene() if vis is None else vis

    if points is not None:
        if points.shape.__len__() == 3:
            points = points[-1]
        if points.shape[-1] == 5:
            points = points[:, 1:]
        add_points(vis, points, colors=points_color, offset=center)

    if boxes is not None:
        if boxes.shape.__len__() == 3:
            boxes = boxes[-1]
        add_boxes(vis, boxes, color=boxes_color, offset=center)

    if key_points is not None:
        if key_points.shape.__len__() == 3:
            key_points = key_points[-1]
        add_keypoint(vis, key_points, color=key_points_color, offset=center)

    if show:
        vis.run()
        vis.destroy_window()


def viz_scenes(*scenes, offset=None, origin=False, title="rd3d"):
    vis = add_scene(origin=origin, title=title)

    if offset is None:
        offset = [0, 0, 0]
    if not isinstance(scenes[0], (list, tuple)):
        scenes = [(scenes[0][i], scenes[1][i]) for i in range(scenes[0].shape[0])]

    num_scenes = len(scenes)
    for i, (points, boxes, *_) in zip(np.linspace(-num_scenes / 2, num_scenes / 2, num_scenes), scenes):
        center = np.array(offset) * i
        viz_scene(points, boxes, *_, center=center, vis=vis)

    vis.run()
    vis.destroy_window()
