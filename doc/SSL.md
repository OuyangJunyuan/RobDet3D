# pipeline
1. 每次迭代训练前先update bank（raw/aug pair predictions）
2. 每个epoch
   1. 对raw点云进行无nms预测，去掉不与bank ins交叠的预测，得到可靠的背景。
   2. 可靠背景经过 ins-bank物体填充 + gt_sampling， 是否应该先填充后加入bank的ins?因为它距离其他未标注instance的gap更小。
   3. （不确定）进行正常的普通的augment
   4. 进行forward和backward
## potential problem
1. currently we use bev iou instead of bev iou 3d.

## kitti_sparse dataset config

```shell
DATASET = dict(
    NAME='KittiDataset',
    TYPE=Path(__file__).stem,
    DATA_PATH=Path('data/kitti_sparse'),
    ...
```

## generate kitti_sparse dataset

```shell
 python tools/ssl/generate_kitti_sparse.py
```

## instance bank analysis

it will cache analysis results under `cache/...`. remove it and rerun following command.

```shell
python - m
rd3d.runner.ss3d.instance_bank
```

# code structures

## runner

在原始训练流程上附加发掘流程: 一共迭代iter_num次

1. 发掘无标注信息
2. 训练`epochs`次
3. 测试1次

```shell
train_flow = ss3d['iter_num'] * [dict(state='mine_miss_anno_ins', split='train', epochs=1),
                                 dict(state='train', split='train', epochs=ss3d['epochs']),
                                 dict(state='test', split='test', epochs=1)]
RUN.workflows.train += train_flow 
```

runner 中从配置文件中导入导入`rd3d.runner.ss3d.ss3d`

```shell
for m in getattr(self, 'custom_import', []): importlib.import_module(m)
```

然后调用`SS3DHook`的runner begine

## mine_miss_anno_ins_one_epoch

当runner的状态切换到`mine_miss_anno_ins`时候会执行这个函数。
该函数在`SS3DHookHelper`中注入到`DistRunnerBase`中以供调用。

```shell
DistRunnerBase.mine_miss_anno_ins_one_epoch = mine_miss_anno_ins_one_epoch
```

### epoch_begin

以mine_miss_anno_ins作为一次迭代学习的第一个状态，因此需要在开始时候设置一下一些量，比如学习率调节器等。

```shell
enable = True
scheduler = build_scheduler
```
---
# NOTE
1. use `torch.multiprocessing.set_sharing_strategy('file_system')` to solve opening too much file for shared tensor when distributed training. 
2. use `pin_memory=False` and `persist_worker=False` to solve `pin_momoery thread exit`
---
# PERFORMANCE
1. iassd 原版进行训练，性能确实有提升。距离全监督差距10点左右
```shell
Car AP@0.70, 0.70, 0.70:
bbox AP:96.9604, 88.6003, 87.8990
bev  AP:89.0813, 86.1749, 82.5590
3d   AP:86.0750, 76.5163, 72.2303
aos  AP:96.80, 88.25, 87.32
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.1406, 91.0205, 88.5483
bev  AP:92.2980, 86.8696, 84.4745
3d   AP:87.2888, 77.6732, 73.4246
aos  AP:97.99, 90.64, 87.96
Car AP@0.70, 0.50, 0.50:
bbox AP:96.9604, 88.6003, 87.8990
bev  AP:97.3696, 89.0233, 88.7045
3d   AP:97.1434, 88.9365, 88.5391
aos  AP:96.80, 88.25, 87.32
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.1406, 91.0205, 88.5483
bev  AP:98.4289, 93.5679, 92.9533
3d   AP:98.3020, 93.3972, 91.0321
aos  AP:97.99, 90.64, 87.96
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:65.1599, 61.3518, 57.2483
bev  AP:56.3932, 51.2807, 47.6147
3d   AP:53.4835, 47.7500, 43.1297
aos  AP:43.22, 41.03, 38.22
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:65.0952, 61.1552, 56.3370
bev  AP:56.0972, 50.4593, 45.6822
3d   AP:52.0690, 46.2626, 41.0684
aos  AP:43.67, 41.41, 38.02
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:65.1599, 61.3518, 57.2483
bev  AP:72.2966, 69.4587, 64.3985
3d   AP:72.2741, 69.4162, 64.2459
aos  AP:43.22, 41.03, 38.22
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:65.0952, 61.1552, 56.3370
bev  AP:73.5216, 69.8618, 64.5455
3d   AP:73.4999, 69.7246, 64.3655
aos  AP:43.67, 41.41, 38.02
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:82.2085, 67.2907, 65.1051
bev  AP:79.5657, 59.4078, 56.1825
3d   AP:73.2335, 56.0505, 52.8685
aos  AP:80.67, 61.72, 59.60
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:83.5928, 68.2030, 64.7724
bev  AP:79.4161, 58.8578, 55.2871
3d   AP:76.0985, 55.8818, 52.4159
aos  AP:81.90, 61.90, 58.74
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:82.2085, 67.2907, 65.1051
bev  AP:82.9231, 66.5602, 63.7768
3d   AP:82.9231, 66.2619, 63.7295
aos  AP:80.67, 61.72, 59.60
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:83.5928, 68.2030, 64.7724
bev  AP:84.2680, 67.0601, 63.8114
3d   AP:84.2663, 66.9447, 63.7440
aos  AP:81.90, 61.90, 58.74
```
pseudo label的召回率比较低，只有28%，但准确度比较高有96%。原因是iou阈值的0.9太苛刻了。考虑动态阈值。
```shell
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        instance bank information                                         │
├────────────┬───────┬──────┬────────┬─────────────────────────────────┬───────────────────────────────────┤
│   class    │   gt  │ anno │ pseudo │  recall [0, 0.0, 0.1, 0.5, 0.7] │ precision [0, 0.0, 0.1, 0.5, 0.7] │
├────────────┼───────┼──────┼────────┼─────────────────────────────────┼───────────────────────────────────┤
│    Car     │ 14191 │ 3001 │  3910  │ [ 0.01  0.00  0.00  0.00  0.34] │  [ 0.03  0.01  0.00  0.01  0.96]  │
│ Pedestrian │  2203 │ 573  │   8    │ [ 0.00  0.00  0.00  0.00  0.00] │  [ 0.25  0.00  0.12  0.12  0.50]  │
│  Cyclist   │  734  │ 138  │   16   │ [ 0.00  0.00  0.00  0.00  0.02] │  [ 0.00  0.00  0.00  0.12  0.88]  │
│    all     │ 17128 │ 3712 │  3934  │ [ 0.01  0.00  0.00  0.00  0.28] │  [ 0.03  0.01  0.00  0.01  0.96]  │
└────────────┴───────┴──────┴────────┴─────────────────────────────────┴───────────────────────────────────┘

```
instance filling 只用最后一个bank的内容。
```shell
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        instance bank information                                         │
├────────────┬───────┬──────┬────────┬─────────────────────────────────┬───────────────────────────────────┤
│   class    │   gt  │ anno │ pseudo │  recall [0, 0.0, 0.1, 0.5, 0.7] │ precision [0, 0.0, 0.1, 0.5, 0.7] │
├────────────┼───────┼──────┼────────┼─────────────────────────────────┼───────────────────────────────────┤
│    Car     │ 14191 │ 3001 │  3985  │ [ 0.01  0.00  0.00  0.00  0.34] │  [ 0.03  0.01  0.00  0.01  0.96]  │
│ Pedestrian │  2203 │ 573  │   8    │ [ 0.00  0.00  0.00  0.00  0.00] │  [ 0.25  0.00  0.12  0.00  0.62]  │
│  Cyclist   │  734  │ 138  │   15   │ [ 0.00  0.00  0.00  0.00  0.02] │  [ 0.07  0.00  0.00  0.07  0.87]  │
│    all     │ 17128 │ 3712 │  4008  │ [ 0.01  0.00  0.00  0.00  0.29] │  [ 0.03  0.01  0.00  0.01  0.96]  │
└────────────┴───────┴──────┴────────┴─────────────────────────────────┴───────────────────────────────────┘
```
# generate 2D semantic mask
```shell
python tools/ssl/generate_2d_ins_masks.py
```