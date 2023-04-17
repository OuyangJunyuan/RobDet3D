# RobDet3D
This repository is deeply inspired by [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d.git), and incorporates their advantages.
In this repo, we develop `HAVSampler` and `GridBallQuery` to speed up `poin-based` model and its deployment.
# INSTALL
```bash
pip install -r requirements.txt
pip install spconv-cuxxx(e.g.cu113)
python setup.py develop
```
# Dataset
refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git) to prepare your data:
```shell
mkdir data && cd data
ln -s path/to/your/kitti/dataset kitti
cd ../
python -m rd3d.datasets.kitti.kitti_dataset \
create_kitti_infos configs/base/datasets/kitti_3cls.py
```

# Training
train your model like this:
```shell
python tools/train.py \
--cfg configs/iassd/iassd_hvcsx1_4x8_80e_sparse_kitti_3cls.py \
wandb --group rd3d
```
or train with multi-GPU
```shell
accelerate launch tools/train.py \
--cfg configs/iassd/iassd_hvcsx1_4x8_80e_sparse_kitti_3cls.py \
wandb --group rd3d
```

# Export TRT Model
```shell
python tools/deploy/export_onnx.py                                      \
--cfg_file configs/iassd/iassd_hvcsx1_4x8_80e_kitti_3cls\(export\).py   \
--ckpt tools/models/iassd_hvcsx1_4x8_80e_kitti_3cls\(export\).pth       \
--onnx tools/models/trt 
``` 
Then the exported onnx model can be found in fold `tools/models/trt `.