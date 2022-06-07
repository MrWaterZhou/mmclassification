checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-4e54846a.pth'
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale='dynamic')
dataset_type = 'PornJson'
model = dict(
    type='ImageClassifier',
    backbone=dict(type='RepVGG', arch='B3g4', out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=2,
        in_channels=2560,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=2,
            mode='multi_label')))
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(
        type='Rotate',
        interpolation='bicubic',
        magnitude_key='angle',
        pad_val=(124, 116, 104),
        magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
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
        pad_val=(124, 116, 104),
        direction='horizontal'),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=(124, 116, 104),
        direction='vertical'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=(124, 116, 104),
        direction='horizontal'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=(124, 116, 104),
        direction='vertical')
]
transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MotionBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='RandomRotate90', p=0.3),
    dict(type='ImageCompression', quality_lower=30, quality_upper=70, p=0.7)
]
transform_after = [dict(type='Cutout', fill_value=128, p=0.3)]
train_pipeline = [
    dict(type='LoadImageFromFileWithLocalAugment', p=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MotionBlur', blur_limit=3, p=1.0)
                ],
                p=0.2),
            dict(type='RandomRotate90', p=0.3),
            dict(
                type='ImageCompression',
                quality_lower=30,
                quality_upper=70,
                p=0.7)
        ]),
    dict(type='RandomResize', size=(240, 240)),
    dict(type='RandomCrop', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(
                type='Rotate',
                interpolation='bicubic',
                magnitude_key='angle',
                pad_val=(124, 116, 104),
                magnitude_range=(0, 30)),
            dict(
                type='Posterize', magnitude_key='bits',
                magnitude_range=(4, 0)),
            dict(
                type='Solarize', magnitude_key='thr',
                magnitude_range=(256, 0)),
            dict(
                type='Contrast',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Brightness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Sharpness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Shear',
                interpolation='bicubic',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                pad_val=(124, 116, 104),
                direction='horizontal'),
            dict(
                type='Shear',
                interpolation='bicubic',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                pad_val=(124, 116, 104),
                direction='vertical'),
            dict(
                type='Translate',
                interpolation='bicubic',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                pad_val=(124, 116, 104),
                direction='horizontal'),
            dict(
                type='Translate',
                interpolation='bicubic',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                pad_val=(124, 116, 104),
                direction='vertical')
        ],
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5),
    dict(type='Albu', transforms=[dict(type='Cutout', fill_value=128, p=0.3)]),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    train=dict(
        type='PornJson',
        data_prefix='',
        classes=['uniform', 'normal'],
        ann_file=[
            'data/uniform/train.txt',
            '/home/zhou/data2/share/uniform_to_check.txt'
        ],
        pipeline=[
            dict(type='LoadImageFromFileWithLocalAugment', p=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MotionBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.2),
                    dict(type='RandomRotate90', p=0.3),
                    dict(
                        type='ImageCompression',
                        quality_lower=30,
                        quality_upper=70,
                        p=0.7)
                ]),
            dict(type='RandomResize', size=(240, 240)),
            dict(type='RandomCrop', size=(224, 224)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(
                        type='Rotate',
                        interpolation='bicubic',
                        magnitude_key='angle',
                        pad_val=(124, 116, 104),
                        magnitude_range=(0, 30)),
                    dict(
                        type='Posterize',
                        magnitude_key='bits',
                        magnitude_range=(4, 0)),
                    dict(
                        type='Solarize',
                        magnitude_key='thr',
                        magnitude_range=(256, 0)),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Shear',
                        interpolation='bicubic',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        pad_val=(124, 116, 104),
                        direction='horizontal'),
                    dict(
                        type='Shear',
                        interpolation='bicubic',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        pad_val=(124, 116, 104),
                        direction='vertical'),
                    dict(
                        type='Translate',
                        interpolation='bicubic',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        pad_val=(124, 116, 104),
                        direction='horizontal'),
                    dict(
                        type='Translate',
                        interpolation='bicubic',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        pad_val=(124, 116, 104),
                        direction='vertical')
                ],
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5),
            dict(
                type='Albu',
                transforms=[dict(type='Cutout', fill_value=128, p=0.3)]),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='PornJson',
        data_prefix='',
        classes=['uniform', 'normal'],
        ann_file='data/uniform/eval.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='PornJson',
        data_prefix='',
        classes=['uniform', 'normal'],
        ann_file='data/uniform/eval.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(
    interval=5,
    metric=['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1'],
    labels=['uniform', 'normal'])
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90, 120, 150, 180])
runner = dict(type='EpochBasedRunner', max_epochs=120)
work_dir = './work_dirs/uniform_repvgg_uncertain'
gpu_ids = range(0, 2)
