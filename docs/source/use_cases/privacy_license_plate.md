# Privacy Protection (License Plate)

## Overview

As organisations collect more data, there is a need to better protect the identities of individuals in public and private places. AI Singapore has developed a solution that performs face anonymisation. This can be used to comply with the General Data Protection Regulation (GDPR) or other data privacy laws.

Our solution automatically detects and blurs vehicles' license plate. This is further elaborated in a [subsequent section](#how-it-works).

## Demo

To try our solution on your computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [privacy_license_plate.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_license_plate.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_privacy_license_plate.yml>
```

## How it Works

There are two main components to license plate anonymisation: 1) license plate detection using AI, and 2) license plate de-identification.

**1. License Plate Detection**

We use an open-source object detection model known as [YoloV4](https://arxiv.org/abs/2004.10934) to identify license plates. This allows the application to identify the locations of the license plates in an image/video feed. The locations of detected license plates are returned as an array of coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each license plate detected. For more information on how to adjust the license plate detector node, check out the [license plate detector configurable parameters](/peekingduck.pipeline.nodes.model.licenseplate.Node).

**2. License Plate De-Identification**

To perform license plate de-identification, the areas bounded by the bounding boxes are blurred using a Gaussian function (Gaussian blur)

## Nodes Used

These are the nodes used in the earlier demo (also in [privacy_license_plate.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_license_plate.yml)):
```
nodes:
- input.recorded:
    input_dir: <path to video with cars>
- model.license_plate_detector:
    model_type: v4tiny
- dabble.fps
- draw.blur_bbox
- draw.legend
- output.screen
```

**1. License Plate Detection Node**

As mentioned, we use a custom-trained Yolov4 model for license plate detection. By default, it uses the Yolov4-tiny model to detect the license plate. For better accuracy, you can change the parameters in the run config declaration to use the Yolov4 model instead.

**2. License Plate De-Identification Nodes**

You can choose to mosaic or blur the detected license plate using the `draw.mosaic_bbox` or `draw.blur_bbox` node in the run config declaration.

**3. Adjusting Nodes**

With regard to the Yolov4 model, some common node configurations that you might want to adjust are:
- `yolo_score_threshold`: The bounding boxes with confidence score less than the specified score threshold are discarded. (default = 0.1)
- `yolo_iou_threshold`: The overlapping bounding boxes above the specified IoU (Intersection over Union) threshold are discarded. (default = 0.3)

In addition, some common node behaviours that you might want to adjust for the mosaic_bbox and blur_bbox nodes are:
- `mosaic_level`: This defines the resolution of a mosaic filter (width x height). The number corresponds to the number of rows and columns used to create a mosaic. For example, the default setting (mosaic_level: 7) creates a 7 x 7 mosaic filter. Increasing the number increases the intensity of pixelation over an area.

- `blur level`: This defines the standard deviation of the Gaussian kernel used in the Gaussian filter. The higher the blur level, the more opaque is the blurring. (default = 7)