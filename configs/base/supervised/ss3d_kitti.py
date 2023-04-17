cache = False
ss3d = dict(
    iter_num=10,
    epochs=6,
    bank=dict(db_info_path='kitti_dbinfos_train.pkl',
              bk_info_path='ss3d/bkinfos_train.pkl',
              pseudo_database_path='ss3d/pseudo_database'),
    global_augments=[
        dict(type='global_flip', prob=1.0, range=[0, 0]),
        dict(type='global_scale', prob=1.0, range=[0.8, 1.2]),
        dict(type='global_rotate', prob=1.0, range=[-0.78539816, 0.78539816]),
    ],
    points_filling_augment=dict(
        visualize=False,
        type='points_filling',
        remove_extra_width=[0.1, 0.1, 0.1],
        pred_infos_path='ss3d/fill_pts_infos_train.pkl'
    ),
    missing_anno_ins_mining=dict(
        visualize=False,
        get_points_func='get_lidar',
        score_threshold_low=0.1,
        score_threshold_high=0.9,
        iou_threshold=0.9,
        cache=cache,
    ),
    reliable_background_mining=dict(
        visualize=False,
        score_threshold=0.01,
        fill_pts_info_path='ss3d/fill_pts_infos_train.pkl'
    ),
    lr=None,
)

train_flow = ss3d['iter_num'] * [dict(state='mine_miss_anno_ins', split='train', epochs=1),
                                 dict(state='train', split='train', epochs=ss3d['epochs']),
                                 dict(state='test', split='test', epochs=1)]
