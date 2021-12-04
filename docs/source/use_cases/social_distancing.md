# Social Distancing

## Overview

To support the fight against COVID-19, AI Singapore developed a solution to encourage individuals to maintain physical distance from each other. This can be used in many places, such as in malls to encourage social distancing in long queues, or in workplaces to ensure employees' well-being. An example of the latter is [HP Inc.](https://aisingapore.org/2020/06/hp-social-distancing/), which collaborated with us to deploy this solution on edge devices in its manufacturing facility in Singapore.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/social_distancing.gif" width="70%">

The most accurate way to measure distance is to use a 3D sensor with depth perception, such as a RGB-D camera or a LiDAR. However, most cameras such as CCTVs and IP cameras usually only produce 2D videos. We developed heuristics that are able to give an approximate measure of physical distance from 2D videos, circumventing this limitation. This is explained in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [social_distancing.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/social_distancing.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_social_distancing.yml>
```

## How it Works

There are two main components to obtain the distance between individuals: 1) human pose estimation using AI; and 2) depth and distance approximation using heuristics.

**1. Human Pose Estimation**

We use an open source human pose estimation model known as [PoseNet](https://arxiv.org/abs/1505.07427) to identify key human skeletal points. This allows the application to identify where individuals are located within the video feed. The coordinates of the various skeletal points will then be used to determine the distance between individuals.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/posenet_demo.gif" width="70%">

**2. Depth and Distance Approximation**

To measure the distance between individuals, we have to estimate the 3D world coordinates from the keypoints in 2D coordinates. To achieve this, we compute the depth (Z) from the XY coordinates using the relationship below:

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/distance_estimation.png" width="70%">


Where:
- Z = depth or distance of scene point from camera
- f = focal length of camera
- y = y position of image point
- Y = y position of scene point


Y<sub>1</sub> - Y<sub>2</sub> is a reference or “ground truth length” that is required to obtain the depth. After numerous experiments, it was decided that the optimal reference length would be the average height of a human torso (height from human hip to center of face). Width was not used as this value has high variance due to the different body angles of an individual while facing the camera.

Once we have the 3D world coordinates of the individuals in the video, we can compare the distances between each pair of individuals and check if they are too close to each other.

## Nodes Used

These are the nodes used in the earlier demo (also in [social_distancing.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/social_distancing.yml)):
```
nodes:
- input.live
- model.posenet
- dabble.keypoints_to_3d_loc:
    focal_length: 1.14
    torso_factor: 0.9
- dabble.check_nearby_objs:
    near_threshold: 1.5
    tag_msg: "TOO CLOSE!"
- dabble.fps
- draw.poses
- draw.tag
- draw.legend
- output.screen
```

**1. Pose Estimation Model**

By default, we are using the PoseNet model with a ResNet backbone for pose estimation. Please take a look at the [benchmarks](../resources/01b_pose_estimation.rst) of pose estimation models that are included in PeekingDuck if you would like to use a different model variation or an alternative model better suited to your use case.

**2. Adjusting Nodes**

Some common node behaviours that you might need to adjust are:
- `focal_length` & `torso_factor`: We calibrated these settings using a Logitech c170 webcam, with 2 individuals of heights about 1.7m. We recommend running a few experiments on your setup and calibrate these accordingly.
- `tag_msg`: The message to show when individuals are too close.
- `near_threshold`: The minimum acceptable distance between 2 individuals, in metres. For example, if the threshold is set at 1.5m, and 2 individuals are standing 2.0m apart, `tag_msg` doesn't show as they are standing further apart than the threshold. The larger this number, the stricter the social distancing.

For more adjustable node behaviours not listed here, check out the [API Reference](/peekingduck.pipeline.nodes).

**3. Using Object Detection (Optional)**

It is possible to use [object detection nodes](../resources/01a_object_detection.rst) instead of pose estimation. To do so, replace the model node accordingly, and replace the node `dabble.keypoints_to_3d_loc` with `dabble.bbox_to_3d_loc`. The reference or “ground truth length” in this case would be the average height of a human, multiplied by a small factor.

You might need to use this approach if running on a resource-limited device such as a Raspberry Pi. In this situation, you'll need to use the lightweight models, and we find that lightweight object detectors are generally better than lightweight pose estimation models in detecting humans.

The trade-off here is that the estimated distance between individuals will be less accurate. This is because for object detectors, the bounding box will be compared with the average height of a human, but the bounding box height decreases if the person is sitting down or bending over.

## Using with Group Size Checker

As part of COVID-19 measures, the Singapore Government has set restrictions on the group sizes of social gatherings. We've developed a [group size checker](https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker/) that checks if the group size limit has been violated.

The nodes for group size checker can be stacked with social distancing, to perform both at the same time. To find out which nodes are used, check out the [readme](./group_size_checking.md) for group size checker.
