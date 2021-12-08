_base_ = [
    '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale='dynamic')
# dataset settings
dataset_type = 'Porn'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='RegNet', arch='regnetx_800mf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=11,
        in_channels=672,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
    ),
    # train_cfg=dict(augments=[
    #     dict(type='BatchMixup', alpha=0.8, num_classes=3, prob=0.5),
    # ])
)

img_norm_cfg = dict(
    # The mean and std are used in PyCls when training RegNets
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)

policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(
        type='Rotate',
        interpolation='bicubic',
        magnitude_key='angle',
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='horizontal'),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='vertical'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='horizontal'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='vertical')
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomJpegCompression'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_prefix='',
        classes=['性感_胸部', '色情_性行为', '色情_女胸', '性感_臀部', '色情_臀部', '性感_腿部特写',
           '色情_女下体', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部'],
        ann_file='/home/zhou/add_disk1/mmcls/data/porn/train_modify.txttrain_2',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/porn/eval',
        classes=['性感_胸部', '色情_性行为', '色情_女胸', '性感_臀部', '色情_臀部', '性感_腿部特写',
           '色情_女下体', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部'],
        ann_file='/home/zhou/add_disk1/mmcls/data/porn/train_modify.txtvalidate_2',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='',
        classes=['性感_胸部', '色情_性行为', '色情_女胸', '性感_臀部', '色情_臀部', '性感_腿部特写',
           '色情_女下体', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部'],
        ann_file='/home/zhou/add_disk1/mmcls/data/porn/train_modify.txtvalidate_2',
        pipeline=test_pipeline))
load_from = 'https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-800MF-4f9d1e8a.pth'
evaluation = dict(interval=5, metric=['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1'])

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
