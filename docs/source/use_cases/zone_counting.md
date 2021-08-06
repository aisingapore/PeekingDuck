# Zone Counting

## Overview

As part of the COVID-19 preventive measures, the Singapore Government has set restrictions on large event gatherings. Guidelines stipulate that large events can be held but attendees should be split into different groups that are of some distance apart and cannot interact with the other groups. Since AI Singapore developed [object counting](./object_counting.md), we further developed a more complex variation for zone counting. Zone counting allows us to create different zones within a single image and count the number of chosen objects detected in each zone. This can be used with CCTVs in malls, shops or event floors for crowd control or to monitor the above mentioned guidelines.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/zone_counting.gif" width="100%" name="zone_counting_gif">

Zone counting is done by looking at the counts of objects detected by the object detection models that fall within the zones specified. As an example, we can count the number of people in the blue and green zones, as per our gif above. This is explained in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [zone_counting.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/zone_counting.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_zone_counting.yml>
```

## How it Works

There are three main components to obtain the zone counts:
1. The detection from the object detection model, which is the bounding boxes.
1. The bottom midpoint of the bounding boxes, derived from the bounding boxes.
1. The zones, which can be set in the zone count dabble configurable parameters.

**1. Object Detection**

We use an open source object detection estimation model known as [Yolov4](https://arxiv.org/abs/2004.10934) and its smaller and faster variant known as Yolov4-tiny to identify the bounding boxes of chosen objects we want to detect. This allows the application to identify where objects are located within the video feed. The location is returned as two (x, y) coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each object detected. For more information in how adjust the yolo node, checkout the [Yolo configurable parameters](/peekingduck.pipeline.nodes.model.yolo.Node).

We can also use the [EfficientDet model](/peekingduck.pipeline.nodes.model.efficientdet.Node) as a more accurate but slower alternative.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/yolo_demo.gif" width="70%">

**2. Bounding Box to Bottom Midpoint**

Given the top-left (x1, y1) and bottom-right (x2, y2) coordinates of every bounding box, can derive the bottom midpoint of each bounding box. This is done by taking the
the lowest y-coordinate (y2), and the midpoint of the x-coordinates (x1 - x2 / 2). This forms our bottom midpoint. We found that using the bottom midpoint is the most efficient way of telling whether something is in the zone specified, as from the usual top-down or angled camera footages of the use cases, the bottom midpoint of the bounding boxes usually corresponds to the point at which the object is located.

**3. Zones**

Zones are created by specifying the (x, y) coordinates of all the corner points that form the area of the zone, either 1) in fractions of the resolution, or 2) in pixel coordinates. These points **must be set clockwise**. As an example, blue zone in the [zone counting gif](#overview) was created but using the following zone:

`[[0, 0], [0.6, 0], [0.6, 1], [0, 1]]`

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/coordinates_explanation.png" width="100%">

Given a resolution of 1280 by 720, these correspond to the top-left of the image, 60% of the length at the top of the image, 60% of the length at the bottom of the image, and the bottom-left of the image. These points in a clockwise direction that form the rectangular blue zone. Zones do not have to be rectangular in shape. It can be any polygonal shape, dictated by the number and position of the (x, y) coordinates set in a zone.

Note that resolution parameter needs to be configured the resolution parameter to that of the image input before using fractions for the (x, y) coordinates.

For finer control over the exact coordinates, the pixelwise coordinates can be used instead. Using the example, the blue zone would be created using the following zone configuration:

`[[0, 0], [768, 0], [768, 720], [0, 720]]`

When using pixelwise coordinates, the resolution is not needed. However, user should check to ensure that the pixel coordinates given fall within the image resolution so that the zone will work as intended.

Elaboration for this adjustment can be found the section below for adjusting nodes.

**4. Zone Counts**

Given the bottom mid-points of all detected objects, we check if the points fall within the area of the specified zones. If it falls inside any zone, an object count is added for that specific zone. This continues until all objects detected are accounted for, which gives the final count of objects in each specified zone.

## Nodes Used

These are the nodes used in the earlier demo (also in [zone_counting.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/zone_counting.yml)):
```
nodes:
- input.live
- model.yolo:
    detect_ids: [0]
- dabble.bbox_to_btm_midpoint
- dabble.zone_count:
    resolution: [1280, 720]
    zones: [
    [[0, 0], [0.6, 0], [0.6, 1], [0, 1]],
    [[0.6, 0], [1, 0], [1, 1], [0.6, 1]]
    ]
- dabble.fps
- draw.bbox
- draw.btm_midpoint
- draw.zones
- draw.legend
- output.screen
```

**1. Object Detection Node**

By default, the node uses the Yolov4-tiny model for object detection, set to detect people. To use more accurate models, you can try the [Yolov4 model](/peekingduck.pipeline.nodes.model.yolo.Node), or the [EfficientDet model](/peekingduck.pipeline.nodes.model.efficientdet.Node) that is included in our repo.

**2. Bottom Midpoint Node**

The bottom midpoint node is called by including `dabble.bbox_to_btm_midpoint` in the run config declaration. This outputs all the bottom midpoints of all detected bounding boxes. The node has no configurable parameters

**3. Zone Counting Node**

The zone counting node is called by including `dabble.zone_count` in the run config declaration. This uses the bottom midpoints of all detected bounding boxes an outputs the number of object counts in each specified zone. The node configurable parameters can be found below.

**4. Adjusting Nodes**

The zone counting detections depend on the configuration set in the object detection models, such as the type of object to detect, etc. As such, please see the [Yolo node documentation](/peekingduck.pipeline.nodes.model.yolo.Node) or the [Efficientdet node documentation](/peekingduck.pipeline.nodes.model.efficientdet.Node) for adjustable behaviours that can influence the result of the zone counting node.

With regards to the zone counting node, some common node behaviours for the zone counting node that you might need to adjust are:
- `resolution`: If you are planning to use fractions to set the coordinates for the area of the zone, the resolution should be set to the image/video/livestream resolution used.
- `zones`: Used to specify the different zones which you would like to set. Each zone coordinates should be set clock-wise in a list. See section on [nodes used](#nodes-used) on how to properly configure multiple zones.

For more adjustable node behaviours not listed here, check out the [API Reference](/peekingduck.pipeline.nodes.model).
