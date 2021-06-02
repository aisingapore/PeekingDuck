# EfficientDet Node

See also [YOLO](https://github.com/aimakerspace/PeekingDuck/blob/docs/models/yolo.md) for alternative object detection node.

## Overview

To facilitate object detection tasks, PeekingDuck offers the EfficientDet node. The EfficientDet model was developed by [Mingxing Tan _et al._](https://arxiv.org/pdf/1911.09070.pdf). The model was trained using the MS COCO (Microsoft Common Objects in Context) dataset and is capable of detecting objects from [80 categories](#Class-IDs-and-names). EfficientDet node has five levels of compound coefficient (0 - 5). A higher compound coefficient will scale up all dimensions of the backbone network width, depth, and input resolution, which results in better performance and slower inference time. The default compound coefficient is _0_ and can be changed to other values, with other configurable options, by following the steps illustrated [here](#Configurable-parameters).

### Input and outputs

The EfficientDet node's input is an image stored as a three-dimensional NumPy array. For live or recorded videos, the input will be a single video frame per inference.

The EfficientDet node's outputs are the bounding boxes' coordinates, classes name and confidence scores of the detected objects. These results are stored in a dictionary and can be accessed using the dictionary keys shown below. Detailed descriptions of the outputs are in the following sub-section.

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

The full list of configurable parameters for the EfficientDet node are listed in [efficientdet.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/model/efficientdet.yml). To change the default parameters, follow the instructions to configure node behaviour in the [main readme](https://github.com/aimakerspace/PeekingDuck). Below is an example of how the EfficientDet node is configured to use a compound coefficient of 4 for 'person', 'aeroplane' and 'cup' detection.

```yaml
# Example: Configure EfficientDet node to detect 'person', 'aeroplane' and 'cup' using a compound coefficient of 4
nodes:
  - input.live
  - model.efficientdet:
      - model_type: 4
      - detect_ids: [0, 4, 41]
  - draw.bbox
  - output.screen
```

The table shown below is a list of commonly adjusted parameters for the EfficientDet node.

| Parameter       | Description                                                                                             | Variables                                                                                    | Default |
| --------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------- |
| model_type      | Specify the compound coefficient to be used in the EfficientDet node.                                   | _0_, _1_, _2_, _3_ or _4_                                                                    | _0_     |
| score_threshold | Bounding box with a confidence score lesser than the specified confidence score threshold is discarded. | _0_ - _1_                                                                                    | _0.3_   |
| detect_ids      | List of object's corresponding class ID to be detected.                                                 | [_x1_, _x2_, ...] where _x1_ and _x2_ are the class IDs. See below for the ID of each class. | [_0_]   |

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

The model weights and inference code are adapted from the work of [xuannianz](https://github.com/xuannianz/EfficientDet).
