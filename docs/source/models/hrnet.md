# HRNet Node

## Overview

To facilitate pose estimation tasks, PeekingDuck offers the High Resolution Net (HRNet) model. The HRNet model is a single-person pose-estimation model, which estimates the spatial locations of the key body joints (keypoints). The original HRNet model was created by [Wang _et al._](https://arxiv.org/pdf/1902.09212.pdf). HRNet requires a separate person detection model prior to keypoints estimation (top-down method). Hence, the HRNet node has to be used in combination with an object detection node, for example, the [YOLO node](https://github.com/aimakerspace/PeekingDuck/blob/dev/docs/source/models/yolo.md):

```yaml
nodes:
  - input.live
  - model.yolo # or other object detection node
  - model.hrnet
  - draw.poses
  - output.screen
```

The HRNet node estimates 17 keypoints and the names of the keypoints are listed in the table [below](#Keypoints'-name-and-number).

### Keypoints' name and number

| Keypoint number | Keypoint name | Keypoint number | Keypoint name  | Keypoint number | Keypoint name | Keypoint number | Keypoint name |
| --------------- | ------------- | --------------- | -------------- | --------------- | ------------- | --------------- | ------------- |
| 0               | nose          | 5               | left shoulder  | 10              | right wrist   | 15              | left ankle    |
| 1               | left eye      | 6               | right shoulder | 11              | left hip      | 16              | right ankle   |
| 2               | right eye     | 7               | left elbow     | 12              | right hip     |                 |               |
| 3               | leftEar       | 8               | right elbow    | 13              | left knee     |                 |               |
| 4               | right ear     | 9               | left wrist     | 14              | right knee    |                 |               |

### Input and outputs

The HRNet node's input is an image stored as a three-dimensional NumPy array. For live or recorded videos, the input will be a single video frame per inference.

The HRNet node's outputs are the keypoints' coordinates, keypoint scores and keypoint connections. These results are stored in a dictionary and can be accessed using the dictionary keys shown below. Detailed descriptions of the outputs are in the following sub-section.

| Name of output         | Dictionary key  |
| ---------------------- | --------------- |
| Keypoints' coordinates | keypoints       |
| Keypoint scores        | keypoint_scores |
| Keypoint connections   | keypoint_conns  |

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

## Configurable parameters

The full list of configurable parameters for the HRNet node are listed in [hrnet.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/model/hrnet.yml). To change the default parameters, follow the instructions to configure node behaviour in the [main readme](https://github.com/aimakerspace/PeekingDuck). Below is an example of how the HRNet node is configured to increase the score threshold.

```yaml
# Example: Configure HRNet node to increase the score threshold
nodes:
  - input.live
  - model.yolo
  - model.hrnet:
      - score_threshold: 0.4
  - draw.poses
  - output.screen
```

The table shown below is a list of commonly adjusted parameters for the HRNet node.

| Parameter       | Description                                                                                                                                                                     | Variables                                                                  | Default                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------- |
| resolution      | The size of the image is resized and padded to before feeding into the model. A larger size will yield higher accuracy while decreasing the speed of inference. When specifying | { height: _int_, width: _int_ }<br> <br>where int must be divisible by 32. | { height: _192_, width: _256_ } |
| score_threshold | Return instance detections that have score greater or equal to the score threshold.                                                                                             | _0_ - _1_                                                                  | _0.1_                           |

## See also

[PoseNet](https://github.com/aimakerspace/PeekingDuck/blob/dev/docs/source/models/posenet.md) for alternative pose estimation node.
