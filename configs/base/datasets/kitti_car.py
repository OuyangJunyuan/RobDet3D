from easydict import EasyDict as dict
from pathlib import Path

DATASET = dict(
    NAME='KittiDataset',
    TYPE=Path(__file__).stem,
    DATA_PATH='data/kitti',
    CLASS_NAMES=['Car'],
    POINT_CLOUD_RANGE=[0, -40, -3, 70.4, 40, 1],

    DATA_SPLIT={'train': 'train',
                'test': 'val'},

    INFO_PATH={'train': ['kitti_infos_train.pkl'],
               'test': ['kitti_infos_val.pkl']},

    GET_ITEM_LIST=["points"],
    FOV_POINTS_ONLY=True,

    DATA_AUGMENTOR=dict(
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
                 LIMIT_WHOLE_SCENE=True
                 ),
            dict(NAME='random_world_flip',
                 ALONG_AXIS_LIST=['x']),
            dict(NAME='random_world_rotation',
                 WORLD_ROT_ANGLE=[-0.78539816, 0.78539816]),
            dict(NAME='random_world_scaling',
                 WORLD_SCALE_RANGE=[0.95, 1.05])
        ]
    ),

    POINT_FEATURE_ENCODING=dict(
        encoding_type='absolute_coordinates_encoding',
        used_feature_list=['x', 'y', 'z', 'intensity'],
        src_feature_list=['x', 'y', 'z', 'intensity'],
    ),

    DATA_PROCESSOR=[
        dict(NAME='mask_points_and_boxes_outside_range',
             REMOVE_OUTSIDE_BOXES=True),
        dict(NAME='shuffle_points',
             SHUFFLE_ENABLED={'train': True, 'test': False}),
        dict(NAME='transform_points_to_voxels',
             VOXEL_SIZE=[0.05, 0.05, 0.1],
             MAX_POINTS_PER_VOXEL=5,
             MAX_NUMBER_OF_VOXELS={'train': 16000, 'test': 40000})
    ],
    metrics=[
        dict(key='Car_3d/moderate_R40', summary='best', goal='maximize', save=True)
    ]
)
