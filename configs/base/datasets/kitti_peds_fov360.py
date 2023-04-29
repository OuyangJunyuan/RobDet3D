from pathlib import Path
from easydict import EasyDict as dict

DATASET = dict(
    NAME='KittiDataset',
    TYPE=Path(__file__).stem,
    DATA_PATH='data/kitti',
    CLASS_NAMES=['Pedestrian'],
    POINT_CLOUD_RANGE=[-30, -30, -30, 30, 30, 30],

    DATA_SPLIT=dict(train='train',
                    test='val'),
    INFO_PATH=dict(train=['kitti_infos_train.pkl'],
                   test=['kitti_infos_val.pkl']),

    GET_ITEM_LIST=["points"],
    FOV_POINTS_ONLY=False,

    DATA_AUGMENTOR=dict(
        DISABLE_AUG_LIST=['placeholder'],
        AUG_CONFIG_LIST=[
            dict(NAME='gt_sampling',
                 USE_ROAD_PLANE=False,
                 DB_INFO_PATH=['kitti_dbinfos_train.pkl'],
                 PREPARE=dict(filter_by_min_points=['Pedestrian:5'],
                              filter_by_difficulty=[-2]),
                 SAMPLE_GROUPS=['Pedestrian: 10'],
                 NUM_POINT_FEATURES=4,
                 DATABASE_WITH_FAKELIDAR=False,
                 REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
                 LIMIT_WHOLE_SCENE=True),
            dict(NAME='random_world_flip',
                 ALONG_AXIS_LIST=['x', 'y']),
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
        dict(key='Pedestrian_3d/moderate_R40', summary='best', goal='maximize', save=True)
    ]
)
