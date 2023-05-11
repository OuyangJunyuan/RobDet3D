# iou_guided_suppression 错误
写成了如下，因此当n非0时就会直接返回，令所有raw_pred都拿去插入instance bank
最终第二轮发掘就找到6000个物体，而且car@R11的性能在77-79。
```python
if n or m == 0:
    return
```
