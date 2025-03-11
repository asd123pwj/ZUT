_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

data_preprocessor = dict(
    size=(512, 512),  
    test_cfg=dict(size_divisor=32)
)
model = dict(
    data_preprocessor=data_preprocessor, 
    decode_head=dict(num_classes=150)
)

cudnn_benchmark = False
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
train_dataloader = dict(batch_size=16)
find_unused_parameters = True