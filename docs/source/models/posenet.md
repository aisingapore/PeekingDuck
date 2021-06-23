# PoseNet Node

## Overview

To facilitate pose estimation tasks, PeekingDuck offers the PoseNet model. The PoseNet model detects human figures and estimates the spatial locations of the key body joints (keypoints). The original PoseNet model was created by the [Google](https://github.com/tensorflow/tfjs-models/tree/master/posenet) team. The PoseNet node is capable of detecting multiple human figures simultaneously per inference and for each detected human figure, 17 keypoints are estimated. The names of the keypoints are listed in the table [below](#Keypoints'-name-and-number). The default backbone architecture is ResNet50 and can be changed to MobileNetV1, with other configurable options, by following the steps illustrated [here](#Configurable-parameters).

### Keypoints' name and number

| Keypoint number | Keypoint name | Keypoint number | Keypoint name  | Keypoint number | Keypoint name | Keypoint number | Keypoint name |
| --------------- | ------------- | --------------- | -------------- | --------------- | ------------- | --------------- | ------------- |
| 0               | nose          | 5               | left shoulder  | 10              | right wrist   | 15              | left ankle    |
| 1               | left eye      | 6               | right shoulder | 11              | left hip      | 16              | right ankle   |
| 2               | right eye     | 7               | left elbow     | 12              | right hip     |                 |               |
| 3               | leftEar       | 8               | right elbow    | 13              | left knee     |                 |               |
| 4               | right ear     | 9               | left wrist     | 14              | right knee    |                 |               |

### Input and outputs

The PoseNet node's input is an image stored as a three-dimensional NumPy array. For live or recorded videos, the input will be a single video frame per inference.

The PoseNet node's outputs are the keypoints' coordinates, keypoint scores, keypoint connections and bounding boxes' coordinates. These results are stored in a dictionary and can be accessed using the dictionary keys shown below. Detailed descriptions of the outputs are in the following sub-section.

| Name of output              | Dictionary key  |
| --------------------------- | --------------- |
| Keypoints' coordinates      | keypoints       |
| Keypoint scores             | keypoint_scores |
| Keypoint connections        | keypoint_conns  |
| Bounding boxes' coordinates | bboxes          |

#### Keypoints' coordinates - "keypoints"

A _N_ by _17_ by _2_ NumPy array where _N_ represents the number of detected human figures, _17_ represents the number of keypoints and _2_ represents the two spatial coordinates of each keypoint. The two coordinates correspond to:

- x: x-coordinate
- y: y-coordinate

If the keypoint has a low confidence score, the coordinates would be "masked" and replaced by "_-1._" as shown in the example below. The order of the Keypoints' coordinates corresponds to the order of "bboxes", "keypoint_scores" and 'keypoint_conns'.

```python
outputs['keypoints'] = np.array([[[x, y],
                                  [x, y],
                                  ...,
                                  [x, y]]])

# example

outputs['keypoints'] = np.array([[[0.58670201,  0.47576586],
                                  [0.60951909,  0.44109605],
                                  ...,
                                  [-1.       , -1.        ]]])
```

#### Keypoint scores - "keypoint_scores"

A _N_ by _17_ by _1_ NumPy array where _N_ represents the number of detected human figures, _17_ represents the number of keypoints and _1_ represents the keypoint scores. The keypoint scores reflect the probability that a keypoint exists in that position. Note that the score is between _0_ and _1_. The order of the confidence scores corresponds to the order of "bboxes", "keypoints" and 'keypoint_conns'.

```python
outputs['keypoint_scores'] = np.array([[float, float, ..., float ]])


# example

outputs['keypoint_scores'] = np.array([[9.93979812e-01, ..., 9.94700551e-01]])
```

#### Keypoint connections - "keypoint_conns"

A _N_ by _17_ by _2<sub>1</sub>_ by _2<sub>2</sub>_ NumPy array where _N_ represents the number of detected human figures and 17 represents the number of keypoints. 2<sub>1</sub> represents the two connecting keypoints if both keypoints are detected while 2<sub>2</sub> represents the two spatial coordinates of each corresponding connecting keypoint. The two coordinates correspond to:

- x: x-coordinate
- y: y-coordinate

The order of the keypoint connections corresponds to the order of "bboxes", "keypoints" and 'keypoint_scores'.

```python
outputs['keypoint_conns'] = np.array([[[[x,y], [x,y]],
                                       [[x,y], [x,y]],
                                        ...,
                                       [[x,y], [x,y]]]])


# example

outputs['keypoint_conns'] = np.array([[[[0.37138409, 0.98567304], [0.55192859, 0.59019476]],
                                       [[0.4532299, 0.58970387], [0.50471611, 0.63052403]],
                                       ...,
                                       [[0.4532299, 0.58970387], [0.50471611, 0.63052403]]]])
```

#### Bounding boxes' coordinates - "bboxes"

A _N_ by _4_ NumPy array, where N represents the number of detected bounding boxes and 4 represents the four coordinates of each bounding box. The coordinates for each bounding box is determined by the coordinates of the outermost keypoints that are not masked for each detected human figure. The four coordinates correspond to:

- x1: top left x-coordinate
- y1: top left y-coordinate
- x2: bottom right x-coordinate
- y2: bottom right y-coordinate

The order of the bounding boxes' coordinates corresponds to the order of "keypoints", "keypoint_scores" and 'keypoint_conns'.

```python
outputs['bboxes'] = np.array([[x1, y1, x2, y2],
                              [x1, y1, x2, y2],
                               ...,
                              [x1, y1, x2, y2]])

# example

outputs['bboxes'] = np.array([[0.30856234 0.12405036 0.8565467  1.]])
```

## Configurable parameters

The full list of configurable parameters for the PoseNet node are listed in [posenet.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/model/posenet.yml). To change the default parameters, follow the instructions to configure node behaviour in the [main readme](https://github.com/aimakerspace/PeekingDuck). Below is an example of how the PoseNet node is configured to load MobileNetv1-101 as the backbone architecture.

```yaml
# Example: Configure PoseNet node to load MobileNetv1-101
nodes:
  - input.live
  - model.posenet:
      - model_type: 101
  - draw.poses
  - output.screen
```

The table shown below is a list of commonly adjusted parameters for the PoseNet node.

| Parameter          | Description                                                                                                                                                                                                                                                                          | Variables                                                                                                            | Default                         |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| model_type         | ResNet50 is the largest and most accurate model, while the MobileNetV1 are a family of smaller and faster models. The suffix of MobileNetV1 represents the number of layers. Example: MobileNetV1-101 has 101 layers. A larger suffix has higher accuracy but with slower inference. | <li>ResNet50: _resnet_</li><li>MobileNetV1-101: _101_</li><li>MobileNetV1-75: _75_</li><li>MobileNetV1-50: _50_</li> | _resnet_                        |
| resolution         | The size of the image is resized and padded to before feeding into the model. A larger size will yield higher accuracy while decreasing the speed of inference.                                                                                                                      | { height: _int_, width: _int_ }                                                                                      | { height: _225_, width: _225_ } |
| max_pose_detection | The maximum number of poses to detect.                                                                                                                                                                                                                                               | _int_                                                                                                                | _10_                            |
| score_threshold    | Return instance detections that have score greater or equal to the score threshold.                                                                                                                                                                                                  | _0_ - _1_                                                                                                            | _0.4_                           |

## Acknowledgements

The model weights are obtained from [Google](https://github.com/tensorflow/tfjs-models/tree/master/posenet) and the inference code is referenced from the work of [Ross Wightman](https://github.com/rwightman/posenet-python).

## See also

[HRNet](https://github.com/aimakerspace/PeekingDuck/blob/dev/docs/source/models/hrnet.md) for alternative pose estimation node.
