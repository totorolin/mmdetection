# RTMDet: An Empirical Study of Designing Real-Time Object Detectors

> [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we aim to design an efficient real-time object detector that exceeds the YOLO series and is easily extensible for many object recognition tasks such as instance segmentation and rotated object detection. To obtain a more efficient model architecture, we explore an architecture that has compatible capacities in the backbone and neck, constructed by a basic building block that consists of large-kernel depth-wise convolutions. We further introduce soft labels when calculating matching costs in the dynamic label assignment to improve accuracy. Together with better training techniques, the resulting object detector, named RTMDet, achieves 52.8% AP on COCO with 300+ FPS on an NVIDIA 3090 GPU, outperforming the current mainstream industrial detectors. RTMDet achieves the best parameter-accuracy trade-off with tiny/small/medium/large/extra-large model sizes for various application scenarios, and obtains new state-of-the-art performance on real-time instance segmentation and rotated object detection. We hope the experimental results can provide new insights into designing versatile real-time object detectors for many object recognition tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208070055-7233a3d8-955f-486a-82da-b714b3c3bbd6.png"/>
</div>

## Results and Models

### Object Detection

|    Model    | size | box AP | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms)<br>RTX3090 | TRT-FP16-Latency(ms)<br>T4 |                   Config                   |                                                                                                                                                Download                                                                                                                                                |
| :---------: | :--: | :----: | :-------: | :------: | :-----------------------------: | :------------------------: | :----------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  RTMDet-s   | 640  |  44.6  |   8.89    |   14.8   |              1.22               |            2.96            |  [config](./rtmdet_s_8xb32-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602.log.json)       |
|  RTMDet-m   | 640  |  49.4  |   24.71   |  39.27   |              1.62               |            6.41            |  [config](./rtmdet_m_8xb32-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220.log.json)       |
|  RTMDet-l   | 640  |  51.5  |   52.3    |  80.23   |              2.44               |           10.32            |  [config](./rtmdet_l_8xb32-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030.log.json)       |
**Note**:

1. We implement a fast training version of RTMDet in [MMYOLO](https://github.com/open-mmlab/mmyolo). Its training speed is **2.6 times faster** and memory requirement is lower! Try it [here](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet)!
2. The inference speed of RTMDet is measured with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and without NMS.
3. For a fair comparison, the config of bbox postprocessing is changed to be consistent with YOLOv5/6/7 after [PR#9494](https://github.com/open-mmlab/mmdetection/pull/9494), bringing about 0.1~0.3% AP improvement.


## Citation

```latex
@misc{lyu2022rtmdet,
      title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
      author={Chengqi Lyu and Wenwei Zhang and Haian Huang and Yue Zhou and Yudong Wang and Yanyi Liu and Shilong Zhang and Kai Chen},
      year={2022},
      eprint={2212.07784},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Visualization

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

## Deployment Tutorial

Here is a basic example of deploy RTMDet with [MMDeploy-1.x](https://github.com/open-mmlab/mmdeploy/tree/1.x).

### Step1. Install MMDeploy

Before starting the deployment, please make sure you install MMDetection and MMDeploy-1.x correctly.

- Install MMDetection, please refer to the [MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).
- Install MMDeploy-1.x, please refer to the [MMDeploy-1.x installation guide](https://mmdeploy.readthedocs.io/en/1.x/get_started.html#installation).

If you want to deploy RTMDet with ONNXRuntime, TensorRT, or other inference engine,
please make sure you have installed the corresponding dependencies and MMDeploy precompiled packages.

### Step2. Convert Model

After the installation, you can enjoy the model deployment journey starting from converting PyTorch model to backend model by running MMDeploy's `tools/deploy.py`.

The detailed model conversion tutorial please refer to the [MMDeploy document](https://mmdeploy.readthedocs.io/en/1.x/02-how-to-run/convert_model.html).
Here we only give the example of converting RTMDet.

MMDeploy supports converting dynamic and static models. Dynamic models support different input shape, but the inference speed is slower than static models.
To achieve the best performance, we suggest converting RTMDet with static setting.

- If you only want to use ONNX, please use [`configs/mmdet/detection/detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmdeploy/blob/1.x/configs/mmdet/detection/detection_onnxruntime_static.py) as the deployment config.
- If you want to use TensorRT, please use [`configs/mmdet/detection/detection_tensorrt_static-640x640.py`](https://github.com/open-mmlab/mmdeploy/blob/1.x/configs/mmdet/detection/detection_tensorrt_static-640x640.py).

If you want to customize the settings in the deployment config for your requirements, please refer to [MMDeploy config tutorial](https://mmdeploy.readthedocs.io/en/1.x/02-how-to-run/write_config.html).

After preparing the deployment config, you can run the `tools/deploy.py` script to convert your model.
Here we take converting RTMDet-s to TensorRT as an example:

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}

# download RTMDet-s checkpoint
wget -P checkpoint https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

# run the command to start model conversion
python tools/deploy.py \
  configs/mmdet/detection/detection_tensorrt_static-640x640.py \
  ${PATH_TO_MMDET}/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
  checkpoint/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth \
  demo/resources/det.jpg \
  --work-dir ./work_dirs/rtmdet \
  --device cuda:0 \
  --show
```

If the script runs successfully, you will see the following files:

```
|----work_dirs
     |----rtmdet
          |----end2end.onnx  # ONNX model
          |----end2end.engine  # TensorRT engine file
```

After this, you can check the inference results with MMDeploy Model Converter API:

```python
from mmdeploy.apis import inference_model

result = inference_model(
  model_cfg='${PATH_TO_MMDET}/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py',
  deploy_cfg='${PATH_TO_MMDEPLOY}/configs/mmdet/detection/detection_tensorrt_static-640x640.py',
  backend_files=['work_dirs/rtmdet/end2end.engine'],
  img='demo/resources/det.jpg',
  device='cuda:0')
```

#### Advanced Setting

To convert the model with TRT-FP16, you can enable the fp16 mode in your deploy config:

```python
# in MMDeploy config
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True  # enable fp16
    ))
```

To reduce the end to end inference speed with the inference engine, we suggest you to adjust the post-processing setting of the model.
We set a very low score threshold during training and testing to achieve better COCO mAP.
However, in actual usage scenarios, a relatively high score threshold (e.g. 0.3) is usually used.

You can adjust the score threshold and the number of detection boxes in your model config according to the actual usage to reduce the time-consuming of post-processing.

```python
# in MMDetection config
model = dict(
    test_cfg=dict(
        nms_pre=1000,  # keep top-k score bboxes before nms
        min_bbox_size=0,
        score_thr=0.3,  # score threshold to filter bboxes
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=100)  # only keep top-100 as the final results.
)
```

### Step3. Inference with SDK

We provide both Python and C++ inference API with MMDeploy SDK.

To use SDK, you need to dump the required info during converting the model. Just add `--dump-info` to the model conversion command:

```shell
python tools/deploy.py \
  configs/mmdet/detection/detection_tensorrt_static-640x640.py \
  ${PATH_TO_MMDET}/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
  checkpoint/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth \
  demo/resources/det.jpg \
  --work-dir ./work_dirs/rtmdet-sdk \
  --device cuda:0 \
  --show \
  --dump-info  # dump sdk info
```

After running the command, it will dump 3 json files additionally for the SDK:

```
|----work_dirs
     |----rtmdet-sdk
          |----end2end.onnx  # ONNX model
          |----end2end.engine  # TensorRT engine file
          # json files for the SDK
          |----pipeline.json
          |----deploy.json
          |----detail.json
```

#### Python API

Here is a basic example of SDK Python API:

```python
from mmdeploy_python import Detector
import cv2

img = cv2.imread('demo/resources/det.jpg')

# create a detector
detector = Detector(model_path='work_dirs/rtmdet-sdk', device_name='cuda', device_id=0)
# run the inference
bboxes, labels, _ = detector(img)
# Filter the result according to threshold
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
  if score < 0.3:
      continue
  # draw bbox
  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
```

#### C++ API

Here is a basic example of SDK C++ API:

```C++
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "mmdeploy/detector.hpp"

int main() {
  const char* device_name = "cuda";
  int device_id = 0;
  std::string model_path = "work_dirs/rtmdet-sdk";
  std::string image_path = "demo/resources/det.jpg";

  // 1. load model
  mmdeploy::Model model(model_path);
  // 2. create predictor
  mmdeploy::Detector detector(model, mmdeploy::Device{device_name, device_id});
  // 3. read image
  cv::Mat img = cv::imread(image_path);
  // 4. inference
  auto dets = detector.Apply(img);
  // 5. deal with the result. Here we choose to visualize it
  for (int i = 0; i < dets.size(); ++i) {
    const auto& box = dets[i].bbox;
    fprintf(stdout, "box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n",
            i, box.left, box.top, box.right, box.bottom, dets[i].label_id, dets[i].score);
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }
  cv::imwrite("output_detection.png", img);
  return 0;
}
```

To build C++ example, please add MMDeploy package in your CMake project as following:

```cmake
find_package(MMDeploy REQUIRED)
target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
```

#### Other languages

- [C# API Examples](https://github.com/open-mmlab/mmdeploy/tree/1.x/demo/csharp)
- [JAVA API Examples](https://github.com/open-mmlab/mmdeploy/tree/1.x/demo/java)

### Deploy RTMDet Instance Segmentation Model

We support RTMDet-Ins ONNXRuntime and TensorRT deployment after [MMDeploy v1.0.0rc2](https://github.com/open-mmlab/mmdeploy/tree/v1.0.0rc2). And its deployment process is almost consistent with the detection model.

#### Step1. Install MMDeploy >= v1.0.0rc2

Please refer to the [MMDeploy-1.x installation guide](https://mmdeploy.readthedocs.io/en/1.x/get_started.html#installation) to install the latest version.
Please remember to replace the pre-built package with the latest version.
The v1.0.0rc2 package can be downloaded from [v1.0.0rc2 release page](https://github.com/open-mmlab/mmdeploy/releases/tag/v1.0.0rc2).

Step2. Convert Model

This step has no difference with the previous tutorial. The only thing you need to change is switching to the RTMDet-Ins deploy config:

- If you want to use ONNXRuntime, please use [`configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py) as the deployment config.
- If you want to use TensorRT, please use [`configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py).

Here we take converting RTMDet-Ins-s to TensorRT as an example:

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}

# download RTMDet-s checkpoint
wget -P checkpoint https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth

# run the command to start model conversion
python tools/deploy.py \
  configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py \
  ${PATH_TO_MMDET}/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py \
  checkpoint/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth \
  demo/resources/det.jpg \
  --work-dir ./work_dirs/rtmdet-ins \
  --device cuda:0 \
  --show
```

If the script runs successfully, you will see the following files:

```
|----work_dirs
     |----rtmdet-ins
          |----end2end.onnx  # ONNX model
          |----end2end.engine  # TensorRT engine file
```

After this, you can check the inference results with MMDeploy Model Converter API:

```python
from mmdeploy.apis import inference_model

result = inference_model(
  model_cfg='${PATH_TO_MMDET}/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py',
  deploy_cfg='${PATH_TO_MMDEPLOY}/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py',
  backend_files=['work_dirs/rtmdet-ins/end2end.engine'],
  img='demo/resources/det.jpg',
  device='cuda:0')
```

### Model Config

In MMDetection's config, we use `model` to set up detection algorithm components. In addition to neural network components such as `backbone`, `neck`, etc, it also requires `data_preprocessor`, `train_cfg`, and `test_cfg`. `data_preprocessor` is responsible for processing a batch of data output by dataloader. `train_cfg`, and `test_cfg` in the model config are for training and testing hyperparameters of the components.Taking RTMDet as an example, we will introduce each field in the config according to different function modules:

```python
model = dict(
    type='RTMDet',  # The name of detector
    data_preprocessor=dict(  # The config of data preprocessor, usually includes image normalization and padding
        type='DetDataPreprocessor',  # The type of the data preprocessor. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.data_preprocessors.DetDataPreprocessor
        mean=[103.53, 116.28, 123.675],  # Mean values used to pre-training the pre-trained backbone models, ordered in R, G, B
        std=[57.375, 57.12, 58.395],  # Standard variance used to pre-training the pre-trained backbone models, ordered in R, G, B
        bgr_to_rgb=False,  # whether to convert image from BGR to RGB
        batch_augments=None),  # Batch-level augmentations
    backbone=dict(  # The config of backbone
        type='CSPNeXt',  # The type of backbone network. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.backbones.CSPNeXt
        arch='P5',  # Architecture of CSPNeXt, from {P5, P6}. Defaults to P5
        expand_ratio=0.5,  # Ratio to adjust the number of channels of the hidden layer. Defaults to 0.5
        deepen_factor=1,  # Depth multiplier, multiply number of blocks in CSP layer by this amount. Defaults to 1.0
        widen_factor=1,  # Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0
        channel_attention=True,  # Whether to add channel attention in each stage. Defaults to True
        norm_cfg=dict(type='SyncBN'),  # Dictionary to construct and config norm layer. Defaults to dict(type=’BN’, requires_grad=True)
        act_cfg=dict(type='SiLU', inplace=True)),  # Config dict for activation layer. Defaults to dict(type=’SiLU’)
    neck=dict(
        type='CSPNeXtPAFPN',  # The type of neck is CSPNeXtPAFPN. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.necks.CSPNeXtPAFPN
        in_channels=[256, 512, 1024],  # Number of input channels per scale
        out_channels=256,  # Number of output channels (used at each scale)
        num_csp_blocks=3,  # Number of bottlenecks in CSPLayer. Defaults to 3
        expand_ratio=0.5,  # Ratio to adjust the number of channels of the hidden layer. Default: 0.5
        norm_cfg=dict(type='SyncBN'),  # Config dict for normalization layer. Default: dict(type=’BN’)
        act_cfg=dict(type='SiLU', inplace=True)),  # Config dict for activation layer. Default: dict(type=’Swish’)
    bbox_head=dict(
        type='RTMDetSepBNHead',  # The type of bbox_head is RTMDetSepBNHead. RTMDetHead with separated BN layers and shared conv layers. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.dense_heads.RTMDetSepBNHead
        num_classes=80,  # Number of categories excluding the background category
        in_channels=256,  # Number of channels in the input feature map
        stacked_convs=2,  # Whether to share conv layers between stages. Defaults to True
        feat_channels=256,  # Feature channels of convolutional layers in the head
        anchor_generator=dict(  # The config of anchor generator
            type='MlvlPointGenerator',  # The methods use MlvlPointGenerator. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/point_generator.py#L92
            offset=0,  # The offset of points, the value is normalized with corresponding stride. Defaults to 0.5
            strides=[8, 16, 32]),  # Strides of anchors in multiple feature levels in order (w, h)
        bbox_coder=dict(type='DistancePointBBoxCoder'),  # Distance Point BBox coder.This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,right) and decode it back to the original. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/distance_point_bbox_coder.py#L9
        loss_cls=dict(  # Config of loss function for the classification branch
            type='QualityFocalLoss',  # Type of loss for classification branch. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.losses.QualityFocalLoss
            use_sigmoid=True,  # Whether sigmoid operation is conducted in QFL. Defaults to True
            beta=2.0,  # The beta parameter for calculating the modulating factor. Defaults to 2.0
            loss_weight=1.0),  #  Loss weight of current loss
        loss_bbox=dict(  # Config of loss function for the regression branch
            type='GIoULoss',  # Type of loss. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.losses.GIoULoss
            loss_weight=2.0),  # Loss weight of the regression branch
        with_objectness=False,  # Whether to add an objectness branch. Defaults to True
        exp_on_reg=True,  # Whether to use .exp() in regression
        share_conv=True,  # Whether to share conv layers between stages. Defaults to True
        pred_kernel_size=1,  # Kernel size of prediction layer. Defaults to 1
        norm_cfg=dict(type='SyncBN'),  # Config dict for normalization layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg=dict(type='SiLU', inplace=True)),  # Config dict for activation layer. Defaults to dict(type='SiLU')
    train_cfg=dict(  # Config of training hyperparameters for ATSS
        assigner=dict(  # Config of assigner
            type='DynamicSoftLabelAssigner',   # Type of assigner. DynamicSoftLabelAssigner computes matching between predictions and ground truth with dynamic soft label assignment. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py#L40
            topk=13),  # Select top-k predictions to calculate dynamic k best matches for each gt. Defaults to 13
        allowed_border=-1,  # The border allowed after padding for valid anchors
        pos_weight=-1,  # The weight of positive samples during training
        debug=False),  # Whether to set the debug mode
    test_cfg=dict(  # Config for testing hyperparameters for ATSS
        nms_pre=30000,  # The number of boxes before NMS
        min_bbox_size=0,  # The allowed minimal box size
        score_thr=0.001,  # Threshold to filter out boxes
        nms=dict(  # Config of NMS in the second stage
            type='nms',  # Type of NMS
            iou_threshold=0.65),  # NMS threshold
        max_per_img=300),  # Max number of detections of each image
)
```
