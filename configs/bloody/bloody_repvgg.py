_base_ = [
    "../_base_/default_runtime.py"
]
fp16 = dict(loss_scale="dynamic")
# dataset settings
dataset_type = "PornJson"

# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type='RepVGG',
        arch='B3g4',
        out_indices=(3, ),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="MultiLabelLinearClsHead",
        num_classes=2,
        in_channels=2560,
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=2, mode='multi_label'),
    ),
    # train_cfg=dict(augments=[
    #     dict(type="BatchMixup", alpha=0.8, num_classes=3, prob=0.5),
    # ])
)

img_norm_cfg = dict(
    # The mean and std are used in PyCls when training RegNets
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)

policies = [
    dict(type="AutoContrast"),
    dict(type="Equalize"),
    dict(
        type="Rotate",
        interpolation="bicubic",
        magnitude_key="angle",
        pad_val=tuple([round(x) for x in img_norm_cfg["mean"][::-1]]),
        magnitude_range=(0, 30)),
    dict(type="Posterize", magnitude_key="bits", magnitude_range=(4, 0)),
    dict(type="Solarize", magnitude_key="thr", magnitude_range=(256, 0)),
    #dict(
    #    type="SolarizeAdd",
    #    magnitude_key="magnitude",
    #    magnitude_range=(0, 110)),
    #dict(
    #    type="ColorTransform",
    #    magnitude_key="magnitude",
    #    magnitude_range=(0, 0.9)),
    dict(type="Contrast", magnitude_key="magnitude", magnitude_range=(0, 0.9)),
    dict(
        type="Brightness", magnitude_key="magnitude",
        magnitude_range=(0, 0.9)),
    dict(
        type="Sharpness", magnitude_key="magnitude", magnitude_range=(0, 0.9)),
    dict(
        type="Shear",
        interpolation="bicubic",
        magnitude_key="magnitude",
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg["mean"][::-1]]),
        direction="horizontal"),
    dict(
        type="Shear",
        interpolation="bicubic",
        magnitude_key="magnitude",
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg["mean"][::-1]]),
        direction="vertical"),
    dict(
        type="Translate",
        interpolation="bicubic",
        magnitude_key="magnitude",
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg["mean"][::-1]]),
        direction="horizontal"),
    dict(
        type="Translate",
        interpolation="bicubic",
        magnitude_key="magnitude",
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg["mean"][::-1]]),
        direction="vertical")
]

transforms = [
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=1.0),
            dict(type="MotionBlur", blur_limit=3, p=1.0),
        ],
        p=0.2),
    dict(
        type="RandomRotate90",
        p=0.3
    ),
    dict(
        type="ImageCompression",
        quality_lower=30,
        quality_upper=70,
        p=0.7,
    ),
]

transform_after = [
    dict(
        type = "Cutout",
        fill_value=128,
        p = 0.3
        )
]

train_pipeline = [
    dict(type="LoadImageFromFileWithLocalAugment", p=0.5),
    dict(type="Albu", transforms=transforms),
    dict(type="RandomResize", size=(240, 240)),
    dict(type="RandomCrop", size=(224, 224)),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(
        type="RandAugment",
        policies=policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5),
    dict(type="Albu", transforms=transform_after),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"])
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(224, 224)),
    # dict(type="CenterCrop", crop_size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"])
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_prefix="",
        classes=["bloody","normal"],
        ann_file=["data/bloody/train.txt"],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix="",
        classes=["bloody","normal"],
        ann_file="data/bloody/eval.txt",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix="",
        classes=["bloody","normal"],
        ann_file="data/bloody/eval.txt",
        pipeline=test_pipeline))
load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-4e54846a.pth"
evaluation = dict(interval=5, metric=["mAP", "CP", "CR", "CF1", "OP", "OR", "OF1"],labels=["bloody","normal"])

# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", step=[30, 60, 90, 120, 150, 180])
runner = dict(type="EpochBasedRunner", max_epochs=120)
