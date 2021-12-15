# Human Detection and Tracking

## Overview

Multi-Object Tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. AI Singapore has developed a solution that performs human detection and tracking in a single model. This application can have a wide range of applications, starting from video surveillance and human computer interaction to robotics

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/human_tracking.gif" width="100%">

Our solution automatically detects and tracks human persons. This is further elaborated in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [human_tracking.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/human_tracking.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_human_tracking.yml>
```

## How it Works

There are two main components to this MOT: 1) human target detection using AI; and 2) corresponding appearance embedding. 

**1. Human Detection**

We use an open source detection model trained on pedestrian detection and person search datasets known as [JDE](https://arxiv.org/abs/1909.12605) to identify human persons. This allows the application to identify the locations of human persons in a video feed. Each of these locations are represented as a pair of (x, y) coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each human person detected. For more information on how adjust the JDE node, checkout the [JDE configurable parameters](/peekingduck.pipeline.nodes.model.jde.Node).

**2. Appearance Embedding Tracking**

To perform tracking, JDE models the training process as a multi-task learning problem with anchor classification, box regression, and embedding learning. The model outputs a "track_id" for each detection based on the appearance embedding learned.

## Nodes Used

These are the nodes used in the earlier demo (also in [human_tracking.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/human_tracking.yml)):
```yaml
nodes:
- input.live
- model.jde
- dabble.fps
- draw.bbox
- draw.tag
- draw.legend
- output.screen
```

**1. JDE Node**

This node employs a single network to *simultaneously* output detection results and the corresponding appearance embeddings of the detected boxes. Therefore JDE stands for Joint Detection and Embedding. More information regarding the model, i.e. research paper and repository can be found [here](https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.model.jde.Node.html).

JDE employs a DarkNet-53 [YOLOv3](https://arxiv.org/abs/1804.02767) as the backbone network for human detection. To learn appearance embeddings, a metric learning algorithm with triplet loss together is used. Observations are assigned to tracklets using the Hungarian algorithm. The Kalman filter is used to smooth the trajectories and predict the locations of previous tracklets in the current frame.

**2. Adjusting Node**

With regard to the JDE model node, some common behaviours that you might want to adjust are:
- `iou_threshold`: This specifies the threshold value for intersection over union of detections (default = 0.5). 
- `score_threshold`: This specifies the threshold values for the detection confidence (default = 0.5). You may want to lower this value to increase the number of detections.
- `nms_threshold`: This specifies the threshold value for non-maximal suppression (default = 0.4). You may want to lower this value to increase the number of detections.
- `min_box_area`: Minimum value for area of detected bounding box. Calculated by width * height.
- `track_buffer`: This value specifies the threshold to remove track if track is lost for more frames than value.

