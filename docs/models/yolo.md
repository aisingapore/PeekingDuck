# YOLO Node

## Overview

To facilitate object detection tasks, PeekingDuck offers a family of “You Only Look Once,” or YOLO models consisting of YOLOv4 and YOLOv4-tiny. The YOLOv4 was developed by [Bochkovskiy et al.](https://arxiv.org/pdf/2004.10934.pdf), while the YOLOv4-tiny was developed by [Jiang et al.](https://arxiv.org/pdf/2011.04244.pdf). Both models were trained using the MS COCO (Microsoft Common Objects in Context) dataset and are capable of detecting objects from [80 categories](#Class-IDs-and-names). The YOLO node uses the YOLOv4-tiny by default and can be changed to using YOLOv4, with other configurable options, by following the steps illustrated [here](#Configurable-parameters).

### Input and outputs

The YOLO node's input is an image stored as a three-dimensional NumPy array. For live or recorded videos, the input will be a single video frame per inference.

The YOLO node's outputs are the bounding boxes' coordinates, classes name and confidence scores of the detected objects. These results are stored in a dictionary and can be accessed using the dictionary keys shown below. Detailed descriptions of the outputs are in the following sub-section.

| Name of output              | Dictionary key |
| --------------------------- | -------------- |
| Bounding boxes' coordinates | bboxes         |
| Classes name                | bbox_labels    |
| Confidence score            | bbox_scores    |

#### Bounding boxes' coordinates - "bboxes"

A N by 4 NumPy array, where N represents the number of detected bounding boxes and 4 represents the four coordinates of each bounding box. The four coordinates correspond to:

- x1: top left x-coordinate
- y1: top left y-coordinate
- x2: bottom right x-coordinate
- y2: bottom right y-coordinate

The order of the bounding boxes' coordinates corresponds to the order of "bbox_labels" and "bbox_scores".

```python
outputs['bboxes'] = np.array([[x1, y1, x2, y2],
                              ...,
                              [x1, y1, x2, y2]])

# example

outputs['bboxes'] = np.array([[0.30856234 0.12405036 0.8565467  1.]])
```

#### Classes name - "bbox_labels"

A NumPy array of the detected objects' classes name. The order of the classes name corresponds to the order of "bboxes" and "bbox_scores".

```python
outputs['bbox_labels'] = np.array([str, str, ..., str])

# example

outputs['bbox_labels'] = np.array(['person'])
```

#### Confidence scores - "bbox_scores"

A NumPy array of the confidence scores of the predicted objects. The order of the confidence scores corresponds to the order of "bboxes" and "bbox_labels". Note that the score is between 0 and 1.

```python
outputs['bbox_scores'] = np.array([float, float, ..., float])

# example

outputs['bbox_scores'] = np.array([0.34761652])
```

## Configurable parameters

The full list of configurable parameters for the YOLO node are listed in [yolo.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/model/yolo.yml). To change the default parameters, follow the instructions to configure node behaviour in the [main readme](https://github.com/aimakerspace/PeekingDuck). Below is an example of how the YOLO node is configured to use YOLOv4 for 'person', 'aeroplane' and 'cup' detection.

```yaml
# Example: Configure YOLO node to detect 'person', 'aeroplane' and 'cup' using YOLOv4
nodes:
  - input.live
  - model.yolo:
      - model_type: v4
      - detect_ids: [0, 4, 41]
  - draw.bbox
  - output.screen
```

The table shown below is a list of commonly adjusted parameters for the YOLO node.

| Parameter                 | Description                                                                                             | Variables                                                                                                            | Default  |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------- |
| model_type                | YOLOv4-tiny has a faster inference speed, while YOLOv4 has a higher precision and accuracy.             | <ul><li>YOLOv4-tiny: _v4tiny_</li><li> YOLOv4: _v4_</li></ul>                                                        | _v4tiny_ |
| max_output_size_per_class | Maximum number of detected instances for each class in an image.                                        | _int_                                                                                                                | _50_     |
| max_total_size            | Maximum total number of detected instances in an image.                                                 | _int_                                                                                                                | _50_     |
| yolo_iou_threshold        | Overlapping bounding boxes above the specified IoU (Intersection over Union) threshold are discarded.   | _0_ - _1_                                                                                                            | _0.5_    |
| yolo_score_threshold      | Bounding box with a confidence score lesser than the specified confidence score threshold is discarded. | _0_ - _1_                                                                                                            | _0.2_    |
| detect_ids                | List of object's corresponding class ID to be detected.                                                 | [*x1*, *x2*, ...] where _x1_ and _x2_ are the class IDs. [See below](#Class-IDs-and-names) for the ID of each class. | [*0*]    |

#### Class IDs and names

| Class ID | Class name    | Class ID | Class name     | Class ID | Class name   | Class ID | Class name   |
| -------- | ------------- | -------- | -------------- | -------- | ------------ | -------- | ------------ |
| 0        | person        | 20       | elephant       | 40       | wine glass   | 60       | dining table |
| 1        | bicycle       | 21       | bear           | 41       | cup          | 61       | toilet       |
| 2        | car           | 22       | zebra          | 42       | fork         | 62       | tv           |
| 3        | motorcycle    | 23       | giraffe        | 43       | knife        | 63       | laptop       |
| 4        | aeroplane     | 24       | backpack       | 44       | spoon        | 64       | mouse        |
| 5        | bus           | 25       | umbrella       | 45       | bowl         | 65       | remote       |
| 6        | train         | 26       | handbag        | 46       | banana       | 66       | keyboard     |
| 7        | truck         | 27       | tie            | 47       | apple        | 67       | cell phone   |
| 8        | boat          | 28       | suitcase       | 48       | sandwich     | 68       | microwave    |
| 9        | traffic light | 29       | frisbee        | 49       | orange       | 69       | oven         |
| 10       | fire hydrant  | 30       | skis           | 50       | broccoli     | 70       | toaster      |
| 11       | stop sign     | 31       | snowboard      | 51       | carrot       | 71       | sink         |
| 12       | parking meter | 32       | sports ball    | 52       | hot dog      | 72       | refrigerator |
| 13       | bench         | 33       | kite           | 53       | pizza        | 73       | book         |
| 14       | bird          | 34       | baseball bat   | 54       | donut        | 74       | clock        |
| 15       | cat           | 35       | baseball glove | 55       | cake         | 75       | vase         |
| 16       | dog           | 36       | skateboard     | 56       | chair        | 76       | scissors     |
| 17       | horse         | 37       | surfboard      | 57       | couch        | 77       | teddy bear   |
| 18       | sheep         | 38       | tennis racket  | 58       | potted plant | 78       | hair drier   |
| 19       | cow           | 39       | bottle         | 59       | bed          | 79       | toothbrush   |

## Acknowledgements

The model weights are trained by [Việt Hùng](https://github.com/hunglc007/tensorflow-yolov4-tflite) and some of the inference code are adapted from the work of [Zihao Zhang](https://github.com/zzh8829/yolov3-tf2).
