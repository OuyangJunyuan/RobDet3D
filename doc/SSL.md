##  kitti_sparse dataset config
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

