import open3d
import numpy as np


def add_points(vis, x, c=None):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(x[:, :3])

    if c is None:
        if x.shape[-1] > 3:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('turbo')
            c = x[:, -1]
            c = c - c.min()
            # c /= c.max()
            c = cmap(c)[:, :3].reshape(-1, 3)  # rgba -> rgb
            pts.colors = open3d.utility.Vector3dVector(c)
        else:
            pts.colors = open3d.utility.Vector3dVector(np.ones((x.shape[0], 3)))
    elif isinstance(c, list):
        pts.colors = open3d.utility.Vector3dVector(np.ones_like(x) * np.array(c).reshape(1, 3))
    else:
        pts.colors = open3d.utility.Vector3dVector(c)
    vis.add_geometry(pts)


def add_keypoint(vis, points, radius=0.05, color=None, n=None):
    for i in range(points.shape[0]):
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.translate(points[i, :3])
        mesh_sphere.compute_vertex_normals()
        if color is None:
            mesh_sphere.paint_uniform_color([0.9, 0.6, 0.2])
        else:
            mesh_sphere.paint_uniform_color(np.clip(color, 0, 1))
        vis.add_geometry(mesh_sphere)


def add_boxes(vis, boxes, labels=None, scores=None, color=(1, 0, 0)):
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
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
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
        if labels is None:
            box_lines.paint_uniform_color(color)
        else:
            box_lines.paint_uniform_color(box_colormap[labels[i]])

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


def viz_scene(points, boxes, center=None, vis=None):
    def check_numpy(*objs):
        return [obj.copy() if isinstance(obj, np.ndarray) else obj.cpu().numpy() for obj in objs]

    points, boxes = check_numpy(points, boxes)
    if points.shape[-1] == 5: points = points[:, 1:]
    xyz, c = points[:, :3], points[:, 3:]

    show = vis is None
    vis = add_scene() if vis is None else vis
    if center is not None:
        xyz, boxes[:, :3] = xyz + center, boxes[:, :3] + center
    points = np.concatenate([xyz, c], axis=-1)
    add_points(vis, points, None)
    add_boxes(vis, boxes)

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
    for i, (points, boxes) in zip(np.linspace(-num_scenes / 2, num_scenes / 2, num_scenes), scenes):
        center = np.array(offset).reshape([1, 3]) * i
        viz_scene(points, boxes, center, vis=vis)

    vis.run()
    vis.destroy_window()
