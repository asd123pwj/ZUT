_base_ = [
    '../0_ZUT_baseline/van_b2.py'
]

model = dict(
    type='ZeroImageEncoderDecoder',
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='ZeroUniformityLoss', loss_weight=1.0),
        ]
        ),
    )