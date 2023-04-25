# 无额外AUG + max_trans_range 1.5 + angle cls 12 + cls weight 1
```shell
2023-04-20 21:13:54,928   INFO  recall_roi_0.3: 0.0
2023-04-20 21:13:54,928   INFO  recall_rcnn_0.3: 1.0
2023-04-20 21:13:54,928   INFO  recall_roi_0.5: 0.0
2023-04-20 21:13:54,928   INFO  recall_rcnn_0.5: 0.7903225806451613
2023-04-20 21:13:54,928   INFO  recall_roi_0.7: 0.0
2023-04-20 21:13:54,928   INFO  recall_rcnn_0.7: 0.016129032258064516
2023-04-20 21:13:54,950   INFO  Pedestrian AP@0.50, 0.50, 0.50:
```
# 无额外AUG + max_trans_range 332 + angle cls 12 + cls weight 1
```shell
2023-04-20 23:05:23,443   INFO  recall_roi_0.3: 0.0
2023-04-20 23:05:23,443   INFO  recall_rcnn_0.3: 1.0
2023-04-20 23:05:23,444   INFO  recall_roi_0.5: 0.0
2023-04-20 23:05:23,444   INFO  recall_rcnn_0.5: 0.8064516129032258
2023-04-20 23:05:23,444   INFO  recall_roi_0.7: 0.0
2023-04-20 23:05:23,444   INFO  recall_rcnn_0.7: 0.0
2023-04-20 23:05:23,465   INFO  Pedestrian AP@0.50, 0.50, 0.50:
```
# 额外AUG + max_trans_range 332 + angle cls 12 + cls weight 1
```shell
2023-04-21 00:08:50,489   INFO  recall_roi_0.3: 0.0
2023-04-21 00:08:50,489   INFO  recall_rcnn_0.3: 1.0
2023-04-21 00:08:50,489   INFO  recall_roi_0.5: 0.0
2023-04-21 00:08:50,489   INFO  recall_rcnn_0.5: 0.8709677419354839
2023-04-21 00:08:50,489   INFO  recall_roi_0.7: 0.0
2023-04-21 00:08:50,490   INFO  recall_rcnn_0.7: 0.16129032258064516
2023-04-21 00:08:50,516   INFO  Pedestrian AP@0.50, 0.50, 0.50:

```

# 额外AUG + max_trans_range 332 + angle cls 12 + cls weight 1 + kiti-pretrain
accelerate-launch tools/train.py --cfg configs/iassd/wy/iassd_hvcsx2_gqx2_4x2_80e_peds.py --experiment kitti_pretrain --batch 2 --pretrain --ckpt tools/models/iassd_hvcsx2_gqx2_4x8_80e_kitti_peds_4x8_80e_kitti_peds\(default\).pth
```shell
2023-04-21 01:56:11,568   INFO  recall_roi_0.3: 0.0
2023-04-21 01:56:11,568   INFO  recall_rcnn_0.3: 1.0
2023-04-21 01:56:11,568   INFO  recall_roi_0.5: 0.0
2023-04-21 01:56:11,568   INFO  recall_rcnn_0.5: 0.8805970149253731
2023-04-21 01:56:11,568   INFO  recall_roi_0.7: 0.0
2023-04-21 01:56:11,568   INFO  recall_rcnn_0.7: 0.22388059701492538
```
---
# 直接用实物的kitti-pretrain得到
```shell
2023-04-26 21:58:14,508   INFO  recall_roi_0.3: 0.0
2023-04-26 21:58:14,508   INFO  recall_rcnn_0.3: 0.821917808219178
2023-04-26 21:58:14,508   INFO  recall_roi_0.5: 0.0
2023-04-26 21:58:14,508   INFO  recall_rcnn_0.5: 0.3287671232876712
2023-04-26 21:58:14,508   INFO  recall_roi_0.7: 0.0
2023-04-26 21:58:14,508   INFO  recall_rcnn_0.7: 0.0
```
# kitti-pretrain 
```shell
2023-04-26 22:48:29,332   INFO  recall_roi_0.3: 0.0
2023-04-26 22:48:29,332   INFO  recall_rcnn_0.3: 0.9726027397260274
2023-04-26 22:48:29,332   INFO  recall_roi_0.5: 0.0
2023-04-26 22:48:29,332   INFO  recall_rcnn_0.5: 0.726027397260274
2023-04-26 22:48:29,333   INFO  recall_roi_0.7: 0.0
2023-04-26 22:48:29,333   INFO  recall_rcnn_0.7: 0.1917808219178082
```