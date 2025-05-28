_base_ = [
    'r50.py'
]

model = dict(
    pretrained='',
    backbone=dict(
        _delete_=True,
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        drop_path_rate=0.3,
        type='VAN',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b3.pth')
        ),
    neck=dict(in_channels=[64, 128, 320, 512]),
)


train_dataloader = dict(batch_size=8)
find_unused_parameters = False