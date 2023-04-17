import easydict
import torch.nn as nn


def build_optimizer(optim_cfg, model):
    import torch.optim as optim
    cfg = optim_cfg.copy()
    optimizer = getattr(optim, cfg.pop('type'))(params=model.parameters(), **cfg)
    return optimizer


def build_scheduler(lr_cfg, optimizer, **kwargs):
    import torch.optim.lr_scheduler as lr_sched
    cfg = lr_cfg.copy()
    for k, v in cfg.items():
        if isinstance(v, str) and v.startswith('$'):
            cfg[k] = kwargs.get(v[2:-1], None)
    lr_scheduler = getattr(lr_sched, cfg.pop('type'))(optimizer=optimizer, **cfg)
    return lr_scheduler


if __name__ == "__main__":
    OPTIMIZATION_CONFIG = easydict.EasyDict(
        type='AdamW',
        lr=0.0025,
        weight_decay=0.01,
        betas=(0.9, 0.99)
    )
    LR_CONFIG = easydict.EasyDict(
        type='OneCycleLR',
        max_lr=OPTIMIZATION_CONFIG.lr,
        total_steps='${total_steps}',
        pct_start=0.4,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10,
        final_div_factor=1e4,
    )

    # LR_WARMUP=False,
    # WARMUP_EPOCH=1
    # GRAD_NORM_CLIP=10,
    linear = nn.Linear(2, 2)
    opt = build_optimizer(OPTIMIZATION_CONFIG, linear)
    schd = build_scheduler(LR_CONFIG, opt, total_steps=100)
    from matplotlib import pyplot as plt

    lrs = []
    for i in range(100):
        lrs.append(schd.get_lr())
        opt.step()
        schd.step()
    plt.plot(lrs)
    plt.show()
