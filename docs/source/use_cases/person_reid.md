# Person Re-Identification

## Overview

As organisations collect and save more data, there can be a need to recognize an individual captured in diverse times and/or locations over several nonoverlapping camera views.

AI Singapore has developed a solution that performs person re-identification. This can be used to solve the problem of matching and identifying people under the scene of single or cross cameras.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/person_reid.gif" width="100%">

Our solution automatically matches a queried image with a person in a video (live or recorded), and adds a tag the matched bounding box of the queried image. This is further elaborated in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [person_reid.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/person_reid.yml) and run PeekingDuck.

```python
> peekingduck run --config_path <path_to_person_reid.yml>
```

## How it Works

There are two main components to person re-identification: 1) object detection using AI; and 2) feature extraction and matching of query to gallery images. 

**1. Object Detection**

Any object detection model incorporated in PeekingDuck that reads an input image and outputs a bounding box can be used for this task. By default the configuration file uses the YOLOv4-tiny model.

**2. Person Re-Identification (ReID)**

This task is performed using a novel deep CNN, termed Omni-Scale Network (OSNet), for omni-scale feature learning in ReID. This is achieved by designing a residual block composed of multiple convolutional feature streams, each detecting features at a certain scale. Importantly, a novel unified aggregation gate is introduced to dynamically fuse multi-scale features with input-dependent channel-wise weights. To efficiently learn spatial-channel correlations and avoid overfitting, the building block uses both pointwise and depthwise convolutions. By stacking such blocks layer-by-layer, OSNet is extremely lightweight and can be trained from scratch on existing ReID benchmarks.

## Nodes Used

These are the nodes used in the earlier demo (also in [person_reid.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/person_reid.yml)):
```yaml
nodes:
- input.recorded:
    input_dir: <path to video to analyse>
- model.yolo
- model.osnet:
	query_root_dir: "<query dir>"
- dabble.fps
- draw.bbox
- draw.tag
- draw.legend
- output.screen
```

**1. Object Detection Node**

As mentioned, we are able to use any object detector model which has persons as an output class to discover people. By default, it uses the YOLOv4-tiny model to detect persons. For better accuracy, you can change the parameters in the run config declaration to use the YOLOv4 model instead.

**2. Person Re-Identification Node**

The ReID model only runs on 1 detected class which is humans. ReID can be used for single or multiple queries of persons to identify. To perform person re-identification, the user has to provide images/screenshots to query against. These images need to be cropped around the person of interest. When the node is initialized, the model extracts features from the given queried image(s) and stores them in a dictionary alongside their respective folder name.

These images must be placed in a folder structure as follows:

```py
reid/
├── person1/
│   ├── img1
├── person2/
│   ├── img1
```

The file path for the ReID folder is used as the `query_root_path` for `model.reid`. From this folder structure, the directories "person1", "person2" will be the object tags that will be drawn on screen for the respective bounding box of each identified person.

**3. Adjusting Nodes**

With regard to the ReID model, some common node behaviours that you might want to adjust are:
- `model_type`: This specifies the type of OSNet model that will be used (default = "osnet"). You may want to change this setting to "osnet_ain" for a model that performs well on generalized domains.
- `query_root_dir`: This specifies the path directory for ReID folder from above.
- `multi_threshold`: This specifies the threshold value for cosine distance matching. Bounding boxes with a cosine distance greater than the specified threshold (default = 0.3) in the final output are discarded. You may want to increase the multi_threshold to increase the number of detections.
