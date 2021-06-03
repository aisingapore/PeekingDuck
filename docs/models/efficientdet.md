# EfficientDet Node

## Overview

To facilitate object detection tasks, PeekingDuck offers the EfficientDet node. The EfficientDet model was developed by [Mingxing Tan _et al._](https://arxiv.org/pdf/1911.09070.pdf). The model was trained using the MS COCO (Microsoft Common Objects in Context) dataset and is capable of detecting objects from [80 categories](#Class-IDs-and-names). EfficientDet node has five levels of compound coefficient (0 - 5). A higher compound coefficient will scale up all dimensions of the backbone network width, depth, and input resolution, which results in better performance but slower inference time. The default compound coefficient is _0_ and can be changed to other values, with other configurable options, by following the steps illustrated [here](#Configurable-parameters).

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
| 0        | person        | 21       | elephant       | 45       | wine glass   | 66       | dining table |
| 1        | bicycle       | 22       | bear           | 46       | cup          | 69       | toilet       |
| 2        | car           | 23       | zebra          | 47       | fork         | 71       | tv           |
| 3        | motorcycle    | 24       | giraffe        | 48       | knife        | 72       | laptop       |
| 4        | aeroplane     | 26       | backpack       | 49       | spoon        | 73       | mouse        |
| 5        | bus           | 27       | umbrella       | 50       | bowl         | 74       | remote       |
| 6        | train         | 30       | handbag        | 51       | banana       | 75       | keyboard     |
| 7        | truck         | 31       | tie            | 52       | apple        | 76       | cell phone   |
| 8        | boat          | 32       | suitcase       | 53       | sandwich     | 77       | microwave    |
| 9        | traffic light | 33       | frisbee        | 54       | orange       | 78       | oven         |
| 10       | fire hydrant  | 34       | skis           | 55       | broccoli     | 79       | toaster      |
| 12       | stop sign     | 35       | snowboard      | 56       | carrot       | 80       | sink         |
| 13       | parking meter | 36       | sports ball    | 57       | hot dog      | 81       | refrigerator |
| 14       | bench         | 37       | kite           | 58       | pizza        | 83       | book         |
| 15       | bird          | 38       | baseball bat   | 59       | donut        | 84       | clock        |
| 16       | cat           | 39       | baseball glove | 60       | cake         | 85       | vase         |
| 17       | dog           | 40       | skateboard     | 61       | chair        | 86       | scissors     |
| 18       | horse         | 41       | surfboard      | 62       | couch        | 87       | teddy bear   |
| 19       | sheep         | 42       | tennis racket  | 63       | potted plant | 88       | hair drier   |
| 20       | cow           | 43       | bottle         | 64       | bed          | 89       | toothbrush   |

## Acknowledgements

The model weights and inference code are adapted from the work of [xuannianz](https://github.com/xuannianz/EfficientDet).

## See also

[YOLO](https://github.com/aimakerspace/PeekingDuck/blob/docs/models/yolo.md) for alternative object detection node.
