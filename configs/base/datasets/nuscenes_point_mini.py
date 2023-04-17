from pathlib import Path
from easydict import EasyDict as dict

DATASET = dict(
    NAME='NuScenesDataset',
    TYPE=Path(__file__).stem,
    VERSION='v1.0-mini',
    DATA_PATH='data/nuscenes',
    CLASS_NAMES=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
    POINT_CLOUD_RANGE=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],

    DATA_SPLIT=dict(train='train',
                    test='val'),

    INFO_PATH=dict(train=['nuscenes_infos_10sweeps_train.pkl'],
                   test=['nuscenes_infos_10sweeps_val.pkl']),

    MAX_SWEEPS=10,
    PRED_VELOCITY=True,
    FILTER_MIN_POINTS_IN_GT=1,
    SET_NAN_VELOCITY_TO_ZEROS=True,

    BALANCED_RESAMPLING=True,

    DATA_AUGMENTOR=dict(
        DISABLE_AUG_LIST=['placeholder'],
        AUG_CONFIG_LIST=[
            dict(NAME='gt_sampling',
                 DB_INFO_PATH=['nuscenes_dbinfos_10sweeps_withvelo.pkl'],
                 PREPARE=dict(
                     filter_by_min_points=[
                         'car:5', 'truck:5', 'construction_vehicle:5', 'bus:5', 'trailer:5',
                         'barrier:5', 'motorcycle:5', 'bicycle:5', 'pedestrian:5', 'traffic_cone:5']
                 ),
                 SAMPLE_GROUPS=[
                     'car:2', 'truck:3', 'construction_vehicle:7', 'bus:4', 'trailer:6',
                     'barrier:2', 'motorcycle:6', 'bicycle:6', 'pedestrian:2', 'traffic_cone:2'],
                 NUM_POINT_FEATURES=5,
                 DATABASE_WITH_FAKELIDAR=False,
                 REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
                 LIMIT_WHOLE_SCENE=True),
            dict(NAME='random_world_flip',
                 ENABLE_PROB=0.5,
                 ALONG_AXIS_LIST=['x', 'y']),
            dict(NAME='random_box_noise',
                 ENABLE_PROB=0.5,
                 SCALE_RANGE=[1.0, 1.0],
                 LOC_NOISE=[1.0, 1.0, 0.0],
                 ROTATION_RANGE=[-1.04719755, 1.04719755], ),
            dict(NAME='random_world_rotation',
                 ENABLE_PROB=0.5,
                 WORLD_ROT_ANGLE=[-0.3925, 0.3925]),
            dict(NAME='random_world_scaling',
                 ENABLE_PROB=0.5,
                 WORLD_SCALE_RANGE=[0.95, 1.05])
        ]
    ),
    POINT_FEATURE_ENCODING=dict(
        encoding_type='absolute_coordinates_encoding',
        used_feature_list=['x', 'y', 'z', 'intensity', 'timestamp'],
        src_feature_list=['x', 'y', 'z', 'intensity', 'timestamp'],
    ),

    DATA_PROCESSOR=[
        dict(NAME='mask_points_and_boxes_outside_range',
             REMOVE_OUTSIDE_BOXES=True),
        dict(NAME='shuffle_points',
             SHUFFLE_ENABLED=dict(train=True, test=False)),
        dict(NAME='sample_points_by_voxel',
             VOXEL_SIZE=[0.1, 0.1, 0.1],
             KEY_FRAME_NUMBER_OF_VOXELS=dict(train=16384, test=16384),
             OTHER_FRAME_NUMBER_OF_VOXELS=dict(train=49152, test=49152))
    ],
    metrics=[
        # dict(key='Car_3d/moderate_R40', summary='best', goal='maximize', save=True),
        # dict(key='Pedestrian_3d/moderate_R40', summary='best', goal='maximize'),
        # dict(key='Cyclist/moderate_R40', summary='best', goal='maximize')
    ]
)
