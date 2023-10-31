from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载COCO注释文件
coco_gt = COCO('/data/xwl/instances_val2017+new.json')

# 加载COCO评估结果文件
coco_dt = coco_gt.loadRes('/home/xiaoweilin/mmdetection/tools/work_dirs/coco_detection/rtmdet_l_box.bbox.json')

# 设置目标检测参数和阈值
iou_thresholds = [0.5]  # 根据COCO官方评估，默认使用0.5的IOU阈值
recalls = [0.5]

# 运行COCO评估，计算指定条件下的mAP
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.params.recThrs = recalls
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 获取mAP
mAP_10 = coco_eval.stats[1]  # 第2个元素是recall为0.1时的mAP

print("mAP@0.1: ", mAP_10)
