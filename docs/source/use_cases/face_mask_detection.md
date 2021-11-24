# Face Mask Detection

## Overview

As part of COVID-19 measures, the Singapore Government has mandated the wearing of face masks in public places. AI Singapore has developed a solution that checks whether or not a person is wearing a face mask. This can be used in places such as in malls or shops to ensure that visitors adhere to the guidelines.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/mask_detection.gif" width="70%">

We have trained a custom YOLOv4 model to detect whether or not a person is wearing a face mask. This is further elaborated in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [face_mask_detection.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/face_mask_detection.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_face_mask_detection.yml>
```

## How it Works

The main component is the detection of face mask using the custom YOLOv4 model.

**Face Mask Detection**

We use an open source object detection model known as [YOLOv4](https://arxiv.org/abs/2004.10934) and its smaller and faster variant known as YOLOv4-tiny to identify the bounding boxes of human faces with and without face masks. This allows the application to identify the locations of faces and their corresponding classes (no_mask = 0 or mask = 1) in a video feed. Each of these locations are represented as a pair of (x, y) coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each human face detected.

The `yolo_face` node detects human faces with and without face masks using the YOLOv4-tiny model by default. The classes are differentiated by the labels and the colours of the bounding boxes when multiple faces are detected. For more information on how adjust the `yolo_face` node, checkout the [`yolo_face` configurable parameters](/peekingduck.pipeline.nodes.model.yolo_face.Node).

## Nodes Used

These are the nodes used in the earlier demo (also in [face_mask_detection.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/face_mask_detection.yml)):
```
nodes:
- input.live
- model.yolo_face
- dabble.fps
- draw.bbox:
    show_labels: True
- draw.legend
- output.screen
```

**1. Face Mask Detection Node**

By default, the node uses the YOLOv4-tiny model for face detection. For better accuracy, you can try the [YOLOv4 model](/peekingduck.pipeline.nodes.model.yolo_face.Node) that is included in our repo.

**2. Adjusting Nodes**

Some common node behaviours that you might want to adjust are:
- `detect_ids`: This specifies the class to be detected where no_mask = 0 and mask = 1. By default, the model detects faces with and without face masks (default = [0,1]).
- `yolo_score_threshold`: This specifies the threshold value. Bounding boxes with confidence score less than the specified confidence score threshold are discarded. You may want to lower the threshold value to increase the number of detections.
