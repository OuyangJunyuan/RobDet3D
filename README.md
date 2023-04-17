# INSTALL
```bash
pip install -r requirements.txt
python setup.py develop
```

# Export TRT Model
```shell
python tools/export.py --cfg_file  ./configs/iassd/iassd_iou_export_4x8_80e_kitti_3cls.py --ckpt ./tools/models/iassd_iou_4x8_80e_kitti_3cls.pth --engine tools/models/trt 
```