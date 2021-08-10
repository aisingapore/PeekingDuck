# Group Size Checking

## Overview

As part of COVID-19 measures, the Singapore Government has set restrictions on the group sizes of social gatherings. AI Singapore has developed a vision-based [group size checker](https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker/) that checks if the group size limit has been violated. This can be used in many places, such as in malls to ensure that visitors adhere to guidelines, or in workplaces to ensure employees' safety.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/group_size_check_2.gif" width="70%">

To check if individuals belong to a group, we check if the physical distance between them is close. The most accurate way to measure distance is to use a 3D sensor with depth perception, such as a RGB-D camera or a LiDAR. However, most cameras such as CCTVs and IP cameras usually only produce 2D videos. We developed heuristics that are able to give an approximate measure of physical distance from 2D videos, circumventing this limitation. This is explained in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [group_size_checking.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/group_size_checking.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_group_size_checking.yml>
```

## How it Works

There are three main components to obtain the distance between individuals: 1) human pose estimation using AI; 2) depth and distance approximation; and 3) linking individuals to groups.

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

**3. Linking Individuals to Groups**

Once we have the 3D world coordinates of the individuals in the video, we can compare the distances between each pair of individuals. If they are close to each other, we assign them to the same group. This is a dynamic connectivity problem and we use the [quick find algorithm](https://regenerativetoday.com/union-find-data-structure-quick-find-algorithm/) to solve it.

## Nodes Used

These are the nodes used in the earlier demo (also in [group_size_checking.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/group_size_checking.yml)):
```
nodes:
- input.live
- model.posenet
- dabble.keypoints_to_3d_loc:
    focal_length: 1.14
    torso_factor: 0.9
- dabble.group_nearby_objs:
    obj_dist_thres: 1.5
- dabble.check_large_groups:
    group_size_thres: 2
- dabble.fps
- draw.poses
- draw.group_bbox_and_tag
- draw.legend
- output.screen
```

**1. Pose Estimation Model**

By default, we are using the PoseNet model with a Resnet backbone for pose estimation. Depending on the device you're using, you might want to switch to the lighter mobilenet backbone, or to a heavier HRnet model for higher accuracy.


**2. Adjusting Nodes**

Some common node behaviours that you might need to adjust are:
- `focal_length` & `torso_factor`: We calibrated these settings using a Logitech c170 webcam, with 2 individuals of heights about 1.7m. We recommend running a few experiments on your setup and calibrate these accordingly.
- `obj_dist_thres`: The maximum distance between 2 individuals, in metres, before they are considered to be part of a group.
- `group_size_thres`: The acceptable group size limit.

For more adjustable node behaviours not listed here, check out the [API reference](/peekingduck.pipeline.nodes.model).

**3. Using Object Detection (Optional)**

It is possible to use object detection nodes such as `model.yolo` instead of pose estimation. To do so, replace the model node accordingly, and replace the node `dabble.keypoints_to_3d_loc` with `dabble.bbox_to_3d_loc`. The reference or “ground truth length” in this case would be the average height of a human, multipled by a small factor.

You might need to use this approach if running on a resource-limited device such as a Raspberry Pi. In this situation, you'll need to use the lightweight models; we find lightweight object detectors are generally better than lightweight pose estimation models in detecting humans.

The trade-off here is that the estimated distance between individuals will be less accurate. This is because for object detectors, the bounding box will be compared with the average height of a human, but the bounding box height decreases if the person is sitting down or bending over.

## Using with Social Distancing

To combat COVID-19, individuals are encouraged to maintain physical distance from each other. We've developed a social distancing tool that checks if individuals are too close to each other.

The nodes for social distancing can be stacked with group size checker, to perform both at the same time. To find out which nodes are used, check out the [readme](./social_distancing.md) for social distancing.

