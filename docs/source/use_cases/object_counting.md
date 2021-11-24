# Object Counting

## Overview

One of the basic task in computer vision is object counting. AI Singapore developed a simple solution built in conjunction with our object detection models. This can be used with CCTVs in malls, shops or factories for crowd control, or other general object counting. For advanced counting, check out [zone counting](zone_counting.md#zone-counting).

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/object_counting.gif" width="100%">

Counting is done by looking at the count of objects detected by the object detection models. For example, we can count the number of people that appear in a video, as shown in the GIF above. This is explained in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [object_counting.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/object_counting.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_object_counting.yml>
```

## How it Works

The main component to obtain the count is the detections from the object detection model, which are the bounding boxes.

**1. Object Detection**

We use an open source object detection estimation model known as [YOLOv4](https://arxiv.org/abs/2004.10934) and its smaller and faster variant known as YOLOv4-tiny to identify the bounding boxes of chosen objects we want to detect. This allows the application to identify where objects are located within the video feed. The location is returned as two (x, y) coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each object detected. For more information in how adjust the `yolo` node, checkout the [`yolo` configurable parameters](/peekingduck.pipeline.nodes.model.yolo.Node).

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/yolo_demo.gif" width="70%">

**2. Object Counting**

To count the number of objects detected, we simply take the sum of the number of bounding boxes detected for the object.

## Nodes Used

These are the nodes used in the earlier demo (also in [object_counting.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/object_counting.yml)):
```
nodes:
- input.live
- model.yolo:
    detect_ids: [0]
- dabble.bbox_count
- dabble.fps
- draw.bbox
- draw.legend
- output.screen
```

**1. Object Detection Node**

By default, the node uses the YOLOv4-tiny model for object detection, set to detect people. Please take a look at the [benchmarks](../resources/01a_object_detection.rst) of object detection models that are included in PeekingDuck if you would like to use a different model variation or an alternative model better suited to your use case.

**2. Object Counting Node**

The object counting node is called by including `dabble.bbox_count` in the run config declaration. This takes the detected bounding boxes and outputs the total count of bounding boxes. The node has no configurable parameters.

**3. Adjusting Nodes**

The object counting node does not have changeable configurations. However, it depends on the configuration set in the object detection models, such as the type of object to detect, etc. For object detection model used in this demo, please see the [`yolo` node documentation](/peekingduck.pipeline.nodes.model.yolo.Node) for adjustable behaviours that can influence the result of the object counting node.

For more adjustable node behaviours not listed here, check out the [API Reference](/peekingduck.pipeline.nodes).

## More Complex Counting Behaviour

We have a more complex variant of object counting that is called zone counting which makes use of the zone count node. It allows for the creation of zones within a single image, and provides separate counts of the chosen objects detected for objects that fall inside the zones created.

For more information, check out the [readme](./zone_counting.md#zone-counting) for zone counting.
