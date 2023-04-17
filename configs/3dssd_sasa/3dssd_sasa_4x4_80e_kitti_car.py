from ..base.datasets.kitti_car import *
from ..base.runtime.adam_onecycle_4_80e import *

DATASET.DATA_AUGMENTOR = dict(
    DISABLE_AUG_LIST=['placeholder'],
    AUG_CONFIG_LIST=[
        dict(NAME='gt_sampling',
             USE_ROAD_PLANE=True,
             DB_INFO_PATH=['kitti_dbinfos_train.pkl'],
             PREPARE=dict(
                 filter_by_min_points=['Car:5'],
                 filter_by_difficulty=[-1]),
             SAMPLE_GROUPS=['Car:15'],
             NUM_POINT_FEATURES=4,
             DATABASE_WITH_FAKELIDAR=False,
             REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
             LIMIT_WHOLE_SCENE=False,
             DISABLE_AT_LAST_N_EPOCH=0
             ),
        dict(NAME='random_world_flip',
             ALONG_AXIS_LIST=['x'],
             ENABLE_PROB=0.5),
        dict(NAME='random_box_noise',
             SCALE_RANGE=[1.0, 1.0],
             LOC_NOISE=[1.0, 1.0, 0.0],
             ROTATION_RANGE=[-1.04719755, 1.04719755],
             ENABLE_PROB=0.5),
        dict(NAME='random_world_rotation',
             WORLD_ROT_ANGLE=[-0.78539816, 0.78539816],
             ENABLE_PROB=0.5),
        dict(NAME='random_world_scaling',
             WORLD_SCALE_RANGE=[0.9, 1.1],
             ENABLE_PROB=0.5)
    ]
)
DATASET.DATA_PROCESSOR = [
    dict(NAME='mask_points_and_boxes_outside_range',
         REMOVE_OUTSIDE_BOXES=True),
    dict(NAME='shuffle_points',
         SHUFFLE_ENABLED={'train': True, 'test': False}),
    dict(NAME='sample_points',
         NUM_POINTS={'train': 16384, 'test': 16384})
]

MODEL = dict(
    NAME='PointVote',
    CLASS_NAMES=DATASET.CLASS_NAMES,
    BACKBONE_3D=dict(
        NAME='GeneralPointNet2MSG',
        ENCODER=
        [
            dict(samplers=[dict(name='d-fps', range=[0, 16384], sample=4096)],
                 groupers=[dict(name='ball', query=dict(radius=[0.0, 0.2], neighbour=32), mlps=[16, 16, 32]),
                           dict(name='ball', query=dict(radius=[0.2, 0.4], neighbour=32), mlps=[16, 16, 32]),
                           dict(name='ball', query=dict(radius=[0.4, 0.8], neighbour=64), mlps=[32, 32, 64])],
                 aggregation=dict(name='cat-mlps', mlps=[64])),
            dict(samplers=[dict(name='s-fps', range=[0, 4096], sample=512, mlps=[32], gamma=1.0,
                                train=dict(target={'set_ignore_flag': True, 'extra_width': [1.0, 1.0, 1.0]},
                                           loss={'weight': 0.01, 'tb_tag': 'sasa_1'})),
                           dict(name='d-fps', range=[0, 4096], sample=512)],
                 groupers=[dict(name='ball', query=dict(radius=[0.0, 0.4], neighbour=32), mlps=[64, 64, 128]),
                           dict(name='ball', query=dict(radius=[0.4, 0.8], neighbour=32), mlps=[64, 64, 128]),
                           dict(name='ball', query=dict(radius=[0.8, 1.6], neighbour=64), mlps=[64, 96, 128])],
                 aggregation=dict(name='cat-mlps', mlps=[128])),
            dict(samplers=[dict(name='s-fps', range=[0, 512], sample=256, mlps=[64], gamma=1.0,
                                train=dict(target={'set_ignore_flag': True, 'extra_width': [1.0, 1.0, 1.0]},
                                           loss={'weight': 0.1, 'tb_tag': 'sasa_2'})),
                           dict(name='d-fps', range=[512, 1024], sample=256)],
                 groupers=[dict(name='ball', query=dict(radius=[0.0, 1.6], neighbour=32), mlps=[128, 128, 256]),
                           dict(name='ball', query=dict(radius=[1.6, 3.2], neighbour=32), mlps=[128, 196, 256]),
                           dict(name='ball', query=dict(radius=[3.2, 4.8], neighbour=64), mlps=[128, 256, 256])],
                 aggregation=dict(name='cat-mlps', mlps=[256])),
        ]),
    POINT_HEAD=dict(
        NAME='PointHeadVotePlus',
        CLASS_AGNOSTIC=False,
        VOTE_SAMPLER=dict(name='select', sample=[0, 256]),
        VOTE_MODULE=
        [
            dict(mlps=[128],
                 max_translation_range=[3.0, 3.0, 2.0],
                 sa=dict(groupers=[dict(name='ball', query=dict(radius=4.8, neighbour=48), mlps=[256, 256, 512]),
                                   dict(name='ball', query=dict(radius=6.4, neighbour=64), mlps=[256, 256, 1024])],
                         aggregation=dict(name='cat-mlps')),
                 train=dict(target={'set_ignore_flag': False, 'extra_width': [0.1, 0.1, 0.1]},
                            loss={'weight': 1.0, 'tb_tag': 'vote_reg_loss'}))
        ],
        SHARED_FC=[512, 256],
        CLS_FC=[128],
        REG_FC=[128],
        BOX_CODER=dict(name='PointBinResidualCoder', use_mean_size=False, angle_bin_num=12),
        TARGET_CONFIG={'method': 'mask', 'gt_central_radius': 10.0},
        LOSS_CONFIG=dict(LOSS_CLS='WeightedBinaryCrossEntropyLoss',
                         LOSS_REG='WeightedSmoothL1Loss',
                         AXIS_ALIGNED_IOU_LOSS_REGULARIZATION=True,
                         CORNER_LOSS_REGULARIZATION=True,
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
                        NMS_THRESH=0.01,
                        NMS_PRE_MAXSIZE=4096,
                        NMS_POST_MAXSIZE=500,
                        MULTI_CLASSES_NMS=False, )
    )
)
RUN.tracker.metrics = DATASET.pop('metrics', [])
