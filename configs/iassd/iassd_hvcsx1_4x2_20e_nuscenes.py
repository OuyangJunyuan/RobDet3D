from ..base.datasets.nuscenes_point import *
from ..base.runtime.adam_onecycle_2_20e import *
DATASET.BALANCED_RESAMPLING=True
MODEL = dict(
    NAME='PointVote',
    CLASS_NAMES=DATASET.CLASS_NAMES,
    BACKBONE_3D=dict(
        NAME='GeneralPointNet2MSG',
        ENCODER=
        [
            dict(samplers=[dict(name='hvcs_v2', sample=16384, voxel=[0.28, 0.28, 0.25])],
                 groupers=[dict(name='ball', query=dict(radius=0.5, neighbour=16), mlps=[16, 16, 32]),
                           dict(name='ball', query=dict(radius=1.0, neighbour=32), mlps=[32, 32, 64])],
                 aggregation=dict(name='cat-mlps', mlps=[64])),
            dict(samplers=[dict(name='d-fps', sample=4096)],
                 groupers=[dict(name='ball', query=dict(radius=1.0, neighbour=16), mlps=[64, 64, 128]),
                           dict(name='ball', query=dict(radius=2.0, neighbour=32), mlps=[64, 96, 128])],
                 aggregation=dict(name='cat-mlps', mlps=[128])),
            dict(samplers=[dict(name='ctr', sample=2048, mlps=[128], class_name=DATASET.CLASS_NAMES,
                                train=dict(target={'extra_width': [0.5, 0.5, 0.5]},
                                           loss={'weight': 1.0, 'tb_tag': 'sasa_1'}))],
                 groupers=[dict(name='ball', query=dict(radius=2.0, neighbour=16), mlps=[128, 128, 256]),
                           dict(name='ball', query=dict(radius=4.0, neighbour=32), mlps=[128, 256, 256])],
                 aggregation=dict(name='cat-mlps', mlps=[256])),
        ]),
    POINT_HEAD=dict(
        NAME='PointHeadVotePlus',
        CLASS_AGNOSTIC=False,
        VOTE_SAMPLER=dict(name='ctr', sample=1024, mlps=[256], class_name=DATASET.CLASS_NAMES,
                          train=dict(target={'extra_width': [0.5, 0.5, 0.5]},
                                     loss={'weight': 1.0, 'tb_tag': 'sasa_2'})),
        VOTE_MODULE=
        [
            dict(mlps=[256, 256],
                 max_translation_range=[8.0, 8.0, 4.0],
                 sa=dict(groupers=[dict(name='ball', query=dict(radius=4.0, neighbour=32), mlps=[256, 256, 512]),
                                   dict(name='ball', query=dict(radius=8.0, neighbour=48), mlps=[256, 512, 1024])],
                         aggregation=dict(name='cat-mlps')),
                 train=dict(target={'set_ignore_flag': False, 'extra_width': [1.0, 1.0, 1.0]},
                            loss={'weight': 1.0, 'tb_tag': 'vote_reg_loss'}))
        ],
        SHARED_FC=[768],
        CLS_FC=[256, 256],
        REG_FC=[256, 256],
        BOX_CODER=dict(name='PointBinResidualCoder', angle_bin_num=12, pred_velo=True,
                       use_mean_size=True, mean_size=[[4.63, 1.97, 1.74],
                                                      [6.93, 2.51, 2.84],
                                                      [6.37, 2.85, 3.19],
                                                      [10.5, 2.94, 3.47],
                                                      [12.29, 2.90, 3.87],
                                                      [0.50, 2.53, 0.98],
                                                      [2.11, 0.77, 1.47],
                                                      [1.70, 0.60, 1.28],
                                                      [0.73, 0.67, 1.77],
                                                      [0.41, 0.41, 1.07]]),
        TARGET_CONFIG={'method': 'mask', 'gt_central_radius': False, 'extra_width': [0.2, 0.2, 0.2]},
        LOSS_CONFIG=dict(LOSS_CLS='WeightedBinaryCrossEntropyLoss',
                         LOSS_REG='WeightedSmoothL1Loss',
                         AXIS_ALIGNED_IOU_LOSS_REGULARIZATION=True,
                         CORNER_LOSS_REGULARIZATION=True,
                         WEIGHTS={'point_cls_weight': 1.0,
                                  'point_offset_reg_weight': 1.0,
                                  'point_angle_cls_weight': 0.2,
                                  'point_angle_reg_weight': 1.0,
                                  'point_iou_weight': 1.0,
                                  'point_corner_weight': 1.0}),
    ),
    POST_PROCESSING=dict(
        EVAL_METRIC='kitti',
        SCORE_THRESH=0.1,
        OUTPUT_RAW_SCORE=False,
        RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
        NMS_CONFIG=dict(NMS_TYPE='nms_gpu',
                        NMS_THRESH=0.2,
                        NMS_PRE_MAXSIZE=4096,
                        NMS_POST_MAXSIZE=500,
                        MULTI_CLASSES_NMS=False, )
    )
)
RUN.tracker.metrics = DATASET.pop('metrics', [])
