auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        "mmpretrain.models",
    ],
)
data_root = "data/DataSet2/"
dataset_type = "CocoDataset"
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=3, save_best="auto", type="CheckpointHook"
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(
        draw=True,
        test_out_dir="work_dirs/images",
        type="DetVisualizationHook",
    ),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
launcher = "none"
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
model = dict(
    backbone=dict(
        arch="large",
        init_cfg=dict(
            checkpoint="https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth",
            prefix="backbone.",
            type="Pretrained",
        ),
        out_indices=(
            3,
            4,
            11,
            16,
        ),
        type="mmpretrain.MobileNetV3",
    ),
    bbox_head=dict(
        center_sampling=True,
        centerness_on_reg=True,
        feat_channels=192,
        in_channels=192,
        loss_bbox=dict(loss_weight=1.0, type="CIoULoss"),
        loss_centerness=dict(
            loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=True
        ),
        loss_cls=dict(
            alpha=0.25, gamma=2.0, loss_weight=1.0, type="FocalLoss", use_sigmoid=True
        ),
        norm_cfg=dict(num_groups=32, type="GN"),
        norm_on_bbox=True,
        num_classes=8,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type="NASFCOSHead",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        add_extra_convs=True,
        conv_cfg=dict(deform_groups=2, type="DCNv2"),
        in_channels=[
            24,
            40,
            112,
            960,
        ],
        norm_cfg=dict(type="BN"),
        num_outs=5,
        out_channels=192,
        start_level=1,
        type="NASFCOS_FPN",
    ),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type="nms"),
        nms_pre=1000,
        score_thr=0.05,
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type="MaxIoUAssigner",
        ),
        debug=False,
        pos_weight=-1,
    ),
    type="NASFCOS",
)
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type="SGD", weight_decay=0.0001),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.001, type="LinearLR"),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type="MultiStepLR",
    ),
]
pretrained = "https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth"
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="data/ORaph8K/test/annotations/annotations.json",
        backend_args=None,
        data_prefix=dict(img="test/images/"),
        data_root="data/ORaph8K/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        1333,
                        800,
                    ),
                ],
                type="RandomChoiceResize",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="data/ORaph8K/test/annotations/annotations.json",
    backend_args=None,
    format_only=False,
    metric=[
        "bbox",
    ],
    type="CocoMetric",
)
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scales=[
            (
                1333,
                800,
            ),
        ],
        type="RandomChoiceResize",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(max_epochs=24, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=4,
    dataset=dict(
        ann_file="data/ORaph8K/train/annotations/annotations.json",
        backend_args=None,
        data_prefix=dict(img="train/images/"),
        data_root="data/ORaph8K/",
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="YOLOXHSVRandomAug"),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        1600,
                        960,
                    ),
                    (
                        1333,
                        640,
                    ),
                    (
                        1333,
                        800,
                    ),
                    (
                        800,
                        800,
                    ),
                    (
                        640,
                        640,
                    ),
                ],
                type="RandomChoiceResize",
            ),
            dict(prob=0.5, type="RandomFlip"),
            dict(type="PackDetInputs"),
        ],
        type="CocoDataset",
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file="data/ORaph8K/val/annotations/annotations.json",
        backend_args=None,
        data_prefix=dict(img="val/images/"),
        data_root="data/ORaph8K/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file="data/ORaph8K/val/annotations/annotations.json",
    backend_args=None,
    format_only=False,
    metric=[
        "bbox",
        "proposal",
    ],
    type="CocoMetric",
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
work_dir = "work_dirs/OR_FCOS"
