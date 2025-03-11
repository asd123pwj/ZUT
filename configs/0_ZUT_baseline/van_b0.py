_base_ = [
    'r50.py'
]

model = dict(
    pretrained='',
    backbone=dict(
        _delete_=True,
        type='VAN',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b0.pth')
        ),
    neck=dict(in_channels=[32, 64, 160, 256]),
)