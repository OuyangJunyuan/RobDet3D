from ..base.datasets.nuscenes_point_mini import *
from ..base.runtime.adam_onecycle_2_20e import *

DATASET.BALANCED_RESAMPLING = True
MODEL = dict(
    NAME='PointVote',
    CLASS_NAMES=DATASET.CLASS_NAMES,
    BACKBONE_3D=dict(
        NAME='GeneralPointNet2MSG',
        ENCODER=
        [
            dict(samplers=[dict(name='d-fps', sample=16384)],
                 groupers=[dict(name='ball', query=dict(radius=[0, 0.5], neighbour=32), mlps=[32, 58, 64]),
                           dict(name='ball', query=dict(radius=[0.5, 1.0], neighbour=64), mlps=[32, 64, 64])],
                 aggregation=dict(name='cat-mlps', mlps=[64])),
            dict(samplers=[dict(name='s-fps', range=[0, 4096], sample=1024, mlps=[32], gamma=1.0,
                                train=dict(target={'set_ignore_flag': True, 'extra_width': [1.0, 1.0, 1.0]},
                                           loss={'weight': 0.001, 'tb_tag': 'sasa_1'})
                                ),
                           dict(name='d-fps', range=[0, 4096], sample=1024),
                           dict(name='d-fps', range=[4096, 16384], sample=2048), ],
                 groupers=[dict(name='ball', query=dict(radius=[0.0, 1.0], neighbour=32), mlps=[64, 96, 128]),
                           dict(name='ball', query=dict(radius=[1.0, 2.0], neighbour=64), mlps=[64, 128, 128])],
                 aggregation=dict(name='cat-mlps', mlps=[128])),
            dict(samplers=[dict(name='s-fps', range=[0, 4096], sample=1024, mlps=[64], gamma=1.0,
                                train=dict(target={'set_ignore_flag': True, 'extra_width': [1.0, 1.0, 1.0]},
                                           loss={'weight': 0.01, 'tb_tag': 'sasa_2'})
                                ),
                           dict(name='d-fps', range=[0, 4096], sample=2048)],
                 groupers=[dict(name='ball', query=dict(radius=[0.0, 2.0], neighbour=32), mlps=[128, 196, 256]),
                           dict(name='ball', query=dict(radius=[2.0, 4.0], neighbour=64), mlps=[128, 256, 256])],
                 aggregation=dict(name='cat-mlps', mlps=[256])),
            dict(samplers=[dict(name='s-fps', range=[0, 3072], sample=1024, mlps=[128], gamma=1.0,
                                train=dict(target={'set_ignore_flag': True, 'extra_width': [1.0, 1.0, 1.0]},
                                           loss={'weight': 0.1, 'tb_tag': 'sasa_3'})
                                ),
                           dict(name='d-fps', range=[0, 3072], sample=1024)],
                 groupers=[dict(name='ball', query=dict(radius=[0.0, 4.0], neighbour=32), mlps=[256, 256, 384]),
                           dict(name='ball', query=dict(radius=[4.0, 8.0], neighbour=64), mlps=[256, 384, 384])],
                 aggregation=dict(name='cat-mlps', mlps=[384])),
        ]),
    POINT_HEAD=dict(
        NAME='PointHeadVotePlus',
        CLASS_AGNOSTIC=False,
        VOTE_SAMPLER=dict(name='select', range=[0, 2048], sample=[0, 1024]),
        VOTE_MODULE=
        [
            dict(mlps=[256, 256],
                 max_translation_range=[8.0, 8.0, 4.0],
                 sa=dict(groupers=[dict(name='ball', query=dict(radius=4.0, neighbour=48), mlps=[384, 512, 512]),
                                   dict(name='ball', query=dict(radius=8.0, neighbour=72), mlps=[384, 512, 1024])],
                         aggregation=dict(name='cat-mlps')),
                 train=dict(target={'set_ignore_flag': False, 'extra_width': [0.1, 0.1, 0.1]},
                            loss={'weight': 1.0, 'tb_tag': 'vote_reg_loss'}))
        ],
        SHARED_FC=[768, 512],
        CLS_FC=[256],
        REG_FC=[256],
        BOX_CODER=dict(name='PointBinResidualCoder', angle_bin_num=12, pred_velo=True, use_mean_size=False),
        TARGET_CONFIG={'method': 'mask', 'gt_central_radius': 15.0},
        LOSS_CONFIG=dict(LOSS_CLS='WeightedBinaryCrossEntropyLoss',
                         LOSS_REG='WeightedSmoothL1Loss',
                         AXIS_ALIGNED_IOU_LOSS_REGULARIZATION=False,
                         CORNER_LOSS_REGULARIZATION=False,
                         WEIGHTS={'point_cls_weight': 1.0,
                                  'point_offset_reg_weight': 1.0,
                                  'point_angle_cls_weight': 1.0,
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
