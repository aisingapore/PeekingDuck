# Social Distancing

## Overview

To support the fight against COVID-19, AI Singapore developed this solution to encourage individuals to maintain physical distance from each other. This can be used in many places, such as in malls to encourage social distancing in long queues, or in workplaces to ensure employees' well-being. An example of the latter is [HP Inc.](https://aisingapore.org/2020/06/hp-social-distancing/), which collaborated with us to deploy this solution on edge devices in its manufacturing facility in Singapore.

<img src="../../images/readme/social_distancing.gif" width="70%">

The most accurate way to measure distance is to use a 3D sensor with depth perception, such as a RGB-D camera or a LiDAR. However, most cameras such as CCTVs and IP cameras usually only produce 2D videos. We developed heuristics that are able to give an approximate measure of physical distance from 2D videos, circumventing this limitation. This is explained in a [subsequent section](#how-it-works).

## Demo

To try our social distancing solution on your own computer: if you haven't done so, follow the [instructions](https://github.com/aimakerspace/PeekingDuck/blob/dev/README.md/#install-and-run-peekingduck) to install PeekingDuck. You'll be able to run a quick demo using by saving this configuration file: [social_distancing.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/social_distancing.yml) and running PeekingDuck.
```
> peekingduck run --config_path <path_to_config>
```

## How it Works

There are two main components to obtain the distance between individuals: 1) human pose estimation AI model; 2) depth and distance approximation heuristic.

**1. Human Pose Estimation**

We use an open source human pose estimation model known as PoseNet to identify key human skeletal points. This allows the application to identify where individuals are located within the video feed. The coordinates of the various skeletal points will then be used to determine the distance between individuals.

<img src="../../images/readme/posenet_demo.gif" width="70%">

**2. Depth and Distance Approximation**

To measure the distance between individuals, we have to convert the keypoints in 2D coordinates to keypoints in 3D world coordinates. To achieve this, we compute the depth (Z) from the XY coordinates from the relationship below:

<img src="../../images/readme/distance_estimation.png" width="70%">


Where:
- Z = depth or distance of scene point from camera
- f = focal length of camera
- y = y position of image point
- Y = y position of scene point


Y<sub>1</sub> - Y<sub>2</sub> is a reference or “ground truth length” that is required to obtain the depth. After numerous experiments, it was decided that the optimal reference length would be the average height of a human torso (height from human hip to center of face). Width was not used as this value has high variance due to the different body angles of an individual while facing the camera.

## Nodes Used

```
nodes:
- input.live
- model.posenet
- heuristic.keypoints_to_3d_loc:
  - focal_length: 1.14
  - torso_factor: 0.9
- heuristic.check_nearby_objs:
  - near_threshold: 2.0
  - tag_msg: "TOO CLOSE!"
- draw.bbox
- draw.poses
- draw.tag
- draw.fps
- output.screen
```