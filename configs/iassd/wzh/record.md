# 用wy的训练出的模型
```shell
python tools/demo.py \
--cfg_file configs/iassd/wzh/iassd_hvcsx2_gqx2_exp.py \
--ckpt tools/models/iassd_hvcsx2_gqx2_4x2_80e_peds_1x2_80e_kitti_peds_fov360\(aug\).pth --scenes 10
```
效果好像不错
# 用kitti预训练模型
```shell
2023-04-21 01:49:17,235   INFO  recall_roi_0.3: 0.0
2023-04-21 01:49:17,235   INFO  recall_rcnn_0.3: 0.76
2023-04-21 01:49:17,235   INFO  recall_roi_0.5: 0.0
2023-04-21 01:49:17,235   INFO  recall_rcnn_0.5: 0.76
2023-04-21 01:49:17,235   INFO  recall_roi_0.7: 0.0
2023-04-21 01:49:17,235   INFO  recall_rcnn_0.7: 0.12
```
# 直接训练
```shell
2023-04-21 11:39:35,153   INFO  recall_roi_0.3: 0.0
2023-04-21 11:39:35,153   INFO  recall_rcnn_0.3: 0.68
2023-04-21 11:39:35,153   INFO  recall_roi_0.5: 0.0
2023-04-21 11:39:35,153   INFO  recall_rcnn_0.5: 0.6
2023-04-21 11:39:35,154   INFO  recall_roi_0.7: 0.0
2023-04-21 11:39:35,154   INFO  recall_rcnn_0.7: 0.0
```
---
# 仿真
### 直接训练
```shell
2023-04-21 13:34:49,362   INFO  recall_roi_0.3: 0.0
2023-04-21 13:34:49,362   INFO  recall_rcnn_0.3: 0.7222222222222222
2023-04-21 13:34:49,362   INFO  recall_roi_0.5: 0.0
2023-04-21 13:34:49,362   INFO  recall_rcnn_0.5: 0.3888888888888889
2023-04-21 13:34:49,362   INFO  recall_roi_0.7: 0.0
2023-04-21 13:34:49,362   INFO  recall_rcnn_0.7: 0.05555555555555555
```
kitti pretrain
```shell
2023-04-21 14:10:02,078   INFO  recall_roi_0.3: 0.0
2023-04-21 14:10:02,078   INFO  recall_rcnn_0.3: 1.0
2023-04-21 14:10:02,078   INFO  recall_roi_0.5: 0.0
2023-04-21 14:10:02,078   INFO  recall_rcnn_0.5: 0.9444444444444444
2023-04-21 14:10:02,078   INFO  recall_roi_0.7: 0.0
2023-04-21 14:10:02,078   INFO  recall_rcnn_0.7: 0.5
```