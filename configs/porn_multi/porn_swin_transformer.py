_base_ = [
    '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale='dynamic')
# dataset settings
dataset_type = 'PornJson'

# model settings

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='small', img_size=224,
        drop_path_rate=0.3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_prefix='',
        classes=['性感_胸部', '色情_女胸', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部', '色情_裸露下体', '性感_腿部特写'],
        ann_file='/home/zhou/projects/mmclassification/data/porn/porn_with_patch/train.v3.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='',
        classes=['性感_胸部', '色情_女胸', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部', '色情_裸露下体', '性感_腿部特写'],
        ann_file='/home/zhou/projects/mmclassification/data/porn/porn_with_patch/eval.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='',
        classes=['性感_胸部', '色情_女胸', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部', '色情_裸露下体', '性感_腿部特写'],
        ann_file='/home/zhou/projects/mmclassification/data/porn/porn_with_patch/eval.txt',
        pipeline=test_pipeline))

evaluation = dict(interval=5, metric=['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1'],
                  labels=['性感_胸部', '色情_女胸', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部', '色情_裸露下体', '性感_腿部特写'])

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 2 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20 * 1252,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=300)
