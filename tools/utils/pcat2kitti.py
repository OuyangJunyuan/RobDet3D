import warnings
from pathlib import Path
import numpy as np
import matplotlib.image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pcat', nargs='+')
parser.add_argument('--kitti')
args = parser.parse_args()
pcat_dir = [Path(p) for p in args.pcat]
kitti_dir = Path(args.kitti)

annotation_format = ["cls",
                     "occluded",
                     "dx", "dy", "dz",
                     "x", "y", "z",
                     "dir",
                     "heading"]

kitti_format = {"cls": '',
                "truncated": 0, "occluded": 0, "dir": 0,
                "box2d_ul_x": 0, "box2d_ul_y": 0, "box2d_br_x": 1, "box2d_br_y": 1,
                "dz": 0, "dy": 0, "dx": 0,
                "x": 0, "y": 0, "z": 0,
                "heading": 0}


########################################################################################################################
def get_valid_frame_id(paths):
    sets = [{file.stem for file in path.iterdir()} for path in paths]
    valid_set = {file.stem for file in paths[0].iterdir()}
    for s in sets[1:]:
        valid_set &= s
    for s, p in zip(sets, paths):
        if s - valid_set:
            warnings.warn("invalid {}: {}".format(p.stem, ", ".join(s - valid_set)))
    return list(valid_set)


def gen_one_pack(annotation_root_dir, kitti_output_dir):
    import time
    from tqdm import tqdm

    kitti_output_dir = Path(kitti_output_dir)
    kitti_image_sets_dir = kitti_output_dir / "ImageSets"
    kitti_training = kitti_output_dir / "training"
    kitti_label_dir = kitti_training / 'label_2'
    kitti_calib_dir = kitti_training / 'calib'
    kitti_image_dir = kitti_training / 'image_2'
    kitti_velodyne_dir = kitti_training / 'velodyne'

    kitti_image_sets_dir.mkdir(parents=True, exist_ok=True)
    kitti_training.mkdir(parents=True, exist_ok=True)
    kitti_label_dir.mkdir(parents=True, exist_ok=True)
    kitti_calib_dir.mkdir(parents=True, exist_ok=True)
    kitti_image_dir.mkdir(parents=True, exist_ok=True)
    kitti_velodyne_dir.mkdir(parents=True, exist_ok=True)

    trg_fid = len(os.listdir(str(kitti_label_dir)))
    print("{} already have {}".format(kitti_output_dir, trg_fid))
    print("start frame_id with {}".format(trg_fid))
    time.sleep(2)

    pcat_box_dir = annotation_root_dir / "_bbox"
    pcat_pcd_dir = annotation_root_dir / "_pcd"
    src_fid = get_valid_frame_id([pcat_box_dir, pcat_pcd_dir])
    src_fid.sort()

    for fid in tqdm(iterable=src_fid):
        # input
        box_file = pcat_box_dir / (str(fid) + ".txt")
        pcd_file = pcat_pcd_dir / (str(fid) + '.pcd')

        # output
        kitti_fid = "%06d" % trg_fid
        label_2_file = kitti_label_dir / ('%s.txt' % kitti_fid)
        calib_file = kitti_calib_dir / ('%s.txt' % kitti_fid)
        image_2_file = kitti_image_dir / ('%s.png' % kitti_fid)
        lidar_file = kitti_velodyne_dir / ('%s.bin' % kitti_fid)

        # read label
        kitti_label_data = []
        with open(box_file, 'r') as f:
            for line in f.readlines():
                kitti_data = dict(kitti_format)
                pcat_data = line.split()
                for val, key in zip(pcat_data, annotation_format):
                    if key == 'cls':
                        kitti_data[key] = val
                    else:
                        if key == 'heading':
                            kitti_data[key] = float(val) / 180 * np.pi
                        else:
                            kitti_data[key] = float(val)
                kitti_label_data.append(kitti_data)
        if len(kitti_label_data) == 0:
            warnings.warn("find empty label files %s" % box_file.__str__())
            continue

        # write label
        with open(str(label_2_file), 'w') as f:
            for obj in kitti_label_data:
                cls = obj['cls']
                occluded = obj['occluded']
                dir_alpha = obj['dir']
                l, w, h = obj['dx'], obj['dy'], obj['dz']
                x, y, z = obj['x'], obj['y'], obj['z']
                yaw = obj['heading']
                #           cls t    o    a   2d(4)         hwl(3)      loc_z0(3) yaw
                label_str = '%s 0 %.4f %.4f 0 0 1 1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' % \
                            (cls,
                             occluded, dir_alpha,
                             h, w, l,
                             -y, -z + h / 2, x, -yaw - np.pi / 2  # 相机坐标系下，3dbox底面中心坐标
                             )
                f.write(label_str)
            f.flush()

        # write calibration
        with open(str(calib_file), 'w') as f:
            Pi = "P%s: 1 0 0.5 0 0 1 0.5 0 0 0 1 0\n"
            R0 = "R0_rect: 1 0 0 0 1 0 0 0 1\n"
            Tr = "%s: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
            # Tr = "%s: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            calib_str = "%s%s%s%s%s%s%s" % (Pi % "0", Pi % "1", Pi % "2", Pi % "3", R0, Tr % "Tr_velo_to_cam",
                                            Tr % "Tr_imu_to_velo")
            f.write(calib_str)
            f.flush()

        # write image
        image_array = np.ones([1, 1])
        matplotlib.image.imsave(image_2_file, image_array)

        # write points
        # with open(pcd_file, 'br') as f:
        #     lines = f.readlines()
        #     filed_num = len(lines[2].split()) - 1
        #     points_str = lines[11:]
        #     print(lines[0])
        #     print(lines[1])
        #     print(lines[2])
        #     print(lines[3])
        #     print(lines[4])
        # print(points_str[0])
        # content = ''
        # for points_str in points_str:
        #     content += ' ' + points_str
        # points = np.fromstring(content, dtype=np.float32, sep=' ').reshape(-1, filed_num)

        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(pcd_file.__str__(), ).points)
        with open(str(lidar_file), 'wb') as f:
            bin_cloud = np.zeros([len(points), 4], dtype=np.float32)
            bin_cloud[:, :3] = points
            bin_cloud.reshape(-1, 4).astype(np.float32)
            bin_cloud.tofile(f)

        trg_fid += 1


def gen_split(dataset_dir):
    split = 'training'
    split_dir = dataset_dir / 'ImageSets'

    sample_id = []
    for filename in os.listdir(os.path.join(dataset_dir, split, "velodyne")):
        sample_id.append(str(filename.split('.')[0]))
    sample_id.sort()
    indices = np.random.permutation(len(sample_id))
    train_indices_end = int(len(indices) * 0.9)
    train_indices = indices[:train_indices_end].tolist()
    val_indices = indices[train_indices_end:].tolist()

    Path(split_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(split_dir, "train.txt"), 'w') as f:
        train_indices.sort()
        for sample in train_indices:
            f.write(sample_id[sample] + '\n')

    with open(os.path.join(split_dir, "val.txt"), 'w') as f:
        val_indices.sort()
        for sample in val_indices:
            f.write(sample_id[sample] + '\n')

    print("train(%d)" % len(train_indices))
    print(train_indices)
    print("valida(%d)" % len(val_indices))
    print(val_indices)


for each_pcat_dir in pcat_dir:
    gen_one_pack(Path(each_pcat_dir), Path(kitti_dir))

gen_split(Path(kitti_dir))
