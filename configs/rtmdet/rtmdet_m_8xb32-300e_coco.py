_base_ = './rtmdet_l_8xb32-300e_coco.py'

model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192))

test_evaluator = dict(
    format_only=True,
    outfile_prefix='./work_dirs/coco_detection/rtmdet_m_val')
