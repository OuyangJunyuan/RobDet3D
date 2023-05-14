# 如何向框架内添加图像和语义？

参考了如下代码：

1. LoGONet
2. BiProDet
3. GD-MAE

总结如下：

## 配置文件

`DATA_SET`中的`GET_ITEM_LIST`
添加了图像、相机和雷达的标定结果、语义标签、二位框。
这里的的gt_boxes是真实标注的，而我们应该用的是pseudo_boxes_2d和pseudo_segmentation_2d

```yaml
GET_ITEM_LIST: [ 'points', 'images', 'calib_matricies', 'segmentation', 'gt_boxes2d' ]
```

## 数据增强

1. BiProDet：在gt sampling 的时候也对图像进行了sampling。
2. GD-MAE：无

## 数据预处理

1. GD-MAE：进行`imrescal`,`imflip`,`imnormlize`，`impad`等。

## 2D backbone/networks

思考，如果只是3D reference points 到图像 上进行feature sample， 那图像的特征可不可以当做2D点来做特征提取？
能不能使用2D的稀疏卷积？ 这样只需要对被sample到的2D点附近提取它的features对把。不然其他的2D features就让费了。
或者将被sample到的点和其附近的像素，当做是2D点云，再跑一个SA？

1. BiProDet：使用的是 PSPNet，一个图像语义分割模型。
2. GD-MAE(GraphRCNN)：IMG_BACKBONE使用DLASeg(CenterNet的backbone 可变性卷积+DLA)，缺点：需要用DCN，会引入新的CUDA算子。
3. LoGONet: backbone用的swin transformer。融合是deformable transformer。
---
# Change I Make
## kitti_dataset.py
`get_item`里增加对应的获取，和边写对应读取mask和label的函数。
在load_data_to_gpu里修改对应的numpy->torch选项。

## dataset.py
collate_batch的时候，增加对`pseudo_masks_2d`,`pseudo_boxes_2d`的处理。
其中`pseudo_masks_2d`放入`["images", "depth_maps"]`中,而`pseudo_boxes_2d`放入`['gt_boxes2d']`
