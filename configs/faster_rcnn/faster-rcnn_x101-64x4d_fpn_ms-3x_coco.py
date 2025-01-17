_base_ = ['../common/ms_3x_coco.py', '../_base_/models/faster-rcnn_r50_fpn.py']
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

test_evaluator = dict(
    format_only=True,
    outfile_prefix='./work_dirs/coco_detection/faster_rcnn_val')
