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
