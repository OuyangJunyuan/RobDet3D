global_augments = [
    dict(type='global_flip', prob=1.0, range=[0, 0]),
    dict(type='global_scale', prob=1.0, range=[0.8, 1.2]),
    dict(type='global_rotate', prob=1.0, range=[-0.78539816, 0.78539816]),
]
ss3d = dict(
    iter_num=10,
    epochs=6,
    lr=None,
    instance_bank=dict(db_info_path='kitti_dbinfos_train.pkl',
                       bk_info_path='ss3d/bkinfos_train.pkl',
                       pseudo_database_path='ss3d/pseudo_database'),
    unlabeled_instance_mining=dict(
        global_augments=global_augments,
        get_points_func='get_lidar',

        score_threshold_low=0.1,
        iou3d_threshold_low=0.1,
        iou2d_threshold=0.8,
        score_threshold_high=0.9,
        iou3d_threshold_high=0.5,

        visualize=False,
        cache=False,
    ),
    reliable_background_mining=dict(
        score_threshold=0.01,

        visualize=False,
        cache=False,
    ),
    instance_filling=dict(
        type='instance_filling',
        remove_extra_width=[0.1, 0.1, 0.1],
        visualize=False
    ),
)

train_flow = ss3d['iter_num'] * [dict(state='mine_miss_anno_ins', split='train', epochs=1),
                                 dict(state='train', split='train', epochs=ss3d['epochs']),
                                 dict(state='test', split='test', epochs=1)]


def add_ss3d(RUN, LR, DATASET):
    # DATASET.GET_ITEM_LIST = ['points', 'calib_matrices', 'pseudo_instances_2d']
    ss3d.update(root_dir=DATASET.DATA_PATH,
                class_names=DATASET.CLASS_NAMES,
                lr=LR)
    RUN.ss3d = ss3d
    RUN.workflows.train += train_flow
    RUN.custom_import += ['rd3d.runner.ss3d.ss3d']  # import to invoke ss3d hook
