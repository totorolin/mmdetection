# Faster R-CNN

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143881188-ab87720f-5059-4b4e-a928-b540fb8fb84d.png" height="300"/>
</div>

## Pre-trained Models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|                           Backbone                            |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                        Config                        |                                                                                                                                                                                        Download                                                                                                                                                                                         |
| :-----------------------------------------------------------: | :-----: | :-----: | :------: | :------------: | :----: | :--------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [X-101-64x4d-FPN](./faster-rcnn_x101-64x4d_fpn_ms-3x_coco.py) | pytorch |   3x    |   10.0   |                |  43.1  | [config](./faster-rcnn_x101-64x4d_fpn_ms-3x_coco.py) |      [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528.log.json)       |

We further finetune some pre-trained models on the COCO subsets, which only contain only a few of the 80 categories.


## Citation

```latex
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```
