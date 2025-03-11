_base_ = [
    'p2t_tiny.py'
]

model = dict(
    pretrained='',
    backbone=dict(
        _delete_=True,
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        type='VAN',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b1.pth')
        ),
    neck=dict(in_channels=[64, 128, 320, 512]),
)