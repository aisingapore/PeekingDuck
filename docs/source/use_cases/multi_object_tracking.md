# Multi Object Tracking

## Overview

Multi Object Tracking (MOT) is the task of detecting unique objects and tracking them as they move across frames in a video.

AI Singapore has developed a vision-based [multi object tracker](https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker/) that tracks multiple moving objects. This tracking capability is to be used in tandem with an object detection model. Objects to track can be, for example, pedestrians on the street, vehicles in the road, sport players on the court, groups of animals and more.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/vehicles_tracking.gif" width="70%">

Currently, there are 2 types of trackers available with PeekingDuck: MOSSE (using OpenCV), and Intersection Over Union (IOU). This is explained in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [object_tracking.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/object_tracking.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_object_tracker.yml>
```

## How it Works

There are three main components to obtain the track of an object: 1) object detection using AI; 2) measure similarity between objects in frames; and 3) recover the identity information based on the similarity measurement between objects across frames. In this section, each tracking algorithm will be explained based on its components.

### 1. Object Detection

The MOT node requires a detected bounding box from an object detector model. To achieve this with PeekingDuck, you may use our open source models such as YOLOv4, EfficientDet, and [PoseNet](https://arxiv.org/abs/1505.07427) (for human detection only) which return detected bounding boxes. This allows the application to identify where each object is located within the video feed. The location is returned as two (x, y) coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each object detected which will then be used to determine a track for each object.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/yolo_demo.gif" width="70%">

### 2. MOSSE (OpenCV)

Minimum Output Sum of Squared Error (MOSSE) uses an adaptive correlation for object tracking which produces stable correlation filters when initialized using a single frame. MOSSE tracker is robust to variations in lighting, scale, pose, and non-rigid deformations. It also detects occlusion based upon the Peak to Sidelobe Ratio (PSR), which enables the tracker to pause and resume where it left off when the object reappears. MOSSE tracker also operates at a higher fps. It is much faster than other models but, not as accurate.

The bounding boxes detected in the first frame are used to initialize a single tracker instance for each detection. The tracker for each bounding box is then updated per frame and is deleted when the tracker fails to find a match overtime.

To account for new detections in a frame, which do not have an associated tracker, we perform an IOU of the new bounding box with previous tracked bounding boxes. Should the IOU exceed a threshold, it is then associated with a current track, otherwise a new instance of a track is initialized for the new bounding box.

### 3. Intersection Over Union (IOU)

With ever increasing performances of object detectors, the basis for a tracker becomes much more reliable. This enables the deployment of much simpler tracking algorithms which can compete with more sophisticated approaches at a fraction of the computational cost. Check out the original paper [here](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf).

This method is based on the assumption that the detector produces a detection per frame for every object to be tracked, i.e. there are none or only few ”gaps” in the detections. Furthermore, it is assumed that detections of an object in consecutive frames have an unmistakably high overlap IOU (intersection-over-union) which is commonly the case when using sufficiently high frame rates.

Authors propose a simple IOU tracker which essentially continues a track by associating the detection with the highest IOU to the last detection in the previous frame if a certain IOU threshold is met. All detections not assigned to an existing track will start a new one.

## Nodes Used

These are the nodes used for performing MOT:
```yaml
nodes:
- input.recorded:   
    input_dir: <path_to_input_video>
- model.yolo:
    model_type: "v4tiny"
- dabble.fps
- dabble.tracking:
    tracking_type: "iou"
- draw.tag
- draw.bbox
- draw.legend
- output.media_writer:
    output_dir: <path_to_output_folder>
```

**1. Object Detection Node**

By default, the node uses the YOLOv4-tiny model for object detection, set to detect people (detect_ids: [0]). To use more accurate models, you can try the [YOLOv4 model](https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.model.yolo.Node.html), or the [EfficientDet model](https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.model.efficientdet.Node.html) that is included in our repo.

**2. Adjusting Nodes**

Some common node behaviours that you might need to adjust are:

- `model_type`: "v4", or "v4tiny" for model.yolo node. "0", "1", "2", "3", or "4" for model.efficientdet node. Either of these models can be used for object detection.
- `detect_ids`: Object class IDs to be detected. View this [link](https://peekingduck.readthedocs.io/en/stable/resources/02_model_indices.html) for each class' indices.
- `tracking_type`: The type of tracking to be used, choose one of: [iou, mosse]

For more adjustable node behaviours not listed here, check out the [API reference](/peekingduck.pipeline.nodes).

## Tracking Analysis

| Tracker        | FPS  | Pros         | Cons                               |
| -------------- | ---- | ------------ | ---------------------------------- |
| MOSSE (OpenCV) | 22   | Fast         | Not the most accurate, sensitive to occlusions |
| IOU            | 17   | Fast, simple | Sensitive to occlusions                         |
