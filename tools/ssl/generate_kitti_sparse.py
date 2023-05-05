import os
import tqdm
import random
import shutil
from pathlib import Path

random.seed(0)
rnd = random.Random()

data_root = Path('data/kitti').absolute()
output_root = data_root.parent / 'kitti_sparse'
output_root.mkdir(parents=True, exist_ok=True)
splits = 'ImageSets'
test_set = 'testing'
training_set = 'training'
items = ['calib', 'image_2', 'image_3', 'planes', 'velodyne']

target_path = output_root / splits
if not target_path.exists():
    shutil.copytree(data_root / splits, target_path)

target_path = output_root / test_set
if not target_path.exists():
    os.symlink(data_root / test_set, output_root / test_set)

(output_root / training_set).mkdir(parents=True, exist_ok=True)
for item in items:
    target_path = output_root / training_set / item
    if not target_path.exists():
        os.symlink(data_root / training_set / item, target_path)

print('process labels')

target_path = output_root / training_set / 'label_2'
target_path.mkdir(exist_ok=True, parents=True)
validation_frames = [line.split('\n')[0] for line in open(data_root / splits / 'val.txt', 'r').readlines()]
training_frames = [line.split('\n')[0] for line in open(data_root / splits / 'train.txt', 'r').readlines()]
for label in tqdm.tqdm(validation_frames):
    shutil.copy(data_root / training_set / f'label_2/{label}.txt',
                output_root / training_set / f'label_2/{label}.txt')

for label in tqdm.tqdm(training_frames):
    annos = open(data_root / training_set / f'label_2/{label}.txt').readlines()
    annos = [anno.split('\n')[0] for anno in annos]
    valid_annos = [anno for anno in annos if anno.split(' ')[0] != 'DontCare']
    valid_annos = valid_annos or annos
    valid_annos = [anno for anno in valid_annos if anno.split(' ')[0] in ['Car', 'Pedestrian', 'Cyclist']]
    valid_annos = valid_annos or annos

    assert valid_annos
    with open(output_root / training_set / f'label_2/{label}.txt', 'w') as f:
        f.write(rnd.choice(valid_annos))

import os
os.system('python -m rd3d.datasets.kitti.kitti_dataset create_kitti_infos configs/base/datasets/kitti_sparse_3cls.py')
