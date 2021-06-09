<div align="center">
    <img src="images/readme/peekingduck.png" width="30%">
    <h1>PeekingDuck</h1>
</div>

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)](https://pypi.org/project/peekingduck/)
[![PyPI version](https://badge.fury.io/py/peekingduck.svg)](https://pypi.org/project/peekingduck/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## What is PeekingDuck?

PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV) inference. It helps to significantly cut down development time when building CV pipelines. The name "PeekingDuck" is a play on these words: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing).

## Install and Run PeekingDuck

1. Install PeekingDuck from [PyPI](https://pypi.org/project/peekingduck/).
    ```
    > pip install peekingduck
    ```
    *Note: if installing on a device with an ARM processor such as a Raspberry Pi, include the `--no-dependencies` flag.*

2. Create a project folder at a convenient location, and initialize a PeekingDuck project.
    ```
    > mkdir <project_dir>
    > cd <project_dir>
    > peekingduck init
    ```
    The following files and folders will be created upon running `peekingduck init`:
    - `run_config.yml` is the main configuration file for PeekingDuck. It currently contains the [default configuration](run_config.yml), and we'll show you how to modify it in a [later section](#changing-nodes-and-settings). 
    - `custom_nodes` is an optional feature that is discussed in a [subsequent section](#create-custom-nodes).
    ```
    <project_dir>
     ├── run_config.yml
     └── src
          └── custom_nodes
    ``` 

3. Run a demo.
    ```
    > peekingduck run
    ```

    If you have a webcam, you should see the demo running live:

    <img src="images/readme/yolo_demo.gif" width="50%">

    The previous command looks for a `run_config.yml` in the current directory. You can also specify the path of a different config file to be used, as follows:
    ```
    > peekingduck run --config_path <path_to_config>
    ```

    Terminate the program by clicking on the output screen and pressing `q`.

4. For more help on how to use PeekingDuck's command line interface, you can use `peekingduck --help`.

## How PeekingDuck Works

**Nodes** are the building blocks of PeekingDuck. Each node is a wrapper for a Python function, and contains information on how other PeekingDuck nodes may interact with it. 

PeekingDuck has 5 types of nodes:

<img src="diagrams/node_types.drawio.svg">

A **pipeline** governs the behavior of a chain of nodes. The diagram below shows the pipeline used in the previous demo. Nodes in a pipeline are called in sequential order, and the output of one node will be the input to another. For example, `input.live` produces "img", which is taken in by `model.yolo`, and `model.yolo` produces "bboxes", which is taken in by `draw.bbox`.

<img src="diagrams/yolo_demo.drawio.svg">

## Changing Nodes and Settings

Earlier on, the `peekingduck init` command created the `run_config.yml` file, which is PeekingDuck's main configuration file and is responsible for:
- Selecting which nodes to include in the pipeline
- Configuring node behaviour

**1. Selecting which nodes to include in the pipeline**:

  - In the earlier object detection demo, `run_config.yml` used the following nodes:
    ```
    nodes:
      - input.live
      - model.yolo
      - draw.bbox
      - output.screen
    ```
  - Now, let's modify it to run a pose estimation demo using the following nodes:
    ```
    nodes:
      - input.live
      - model.posenet
      - draw.poses
      - output.screen
    ```

  - If you have a webcam, you should see the demo running live:

    <img src="images/readme/posenet_demo.gif" width="50%">

    Terminate the program by clicking on the output screen and pressing `q`.

**2. Configuring node behaviour**:
- If you're not using a webcam, don't worry about missing out! PeekingDuck is also able to work on recorded videos or saved images, and we'll use the `input.recorded` and `output.media_writer` nodes for that. You can use any video or image file as long as it's a supported format, or [download](https://peekingduck.blob.core.windows.net/videos/running.mp4.zip) a short sample video (credit: [PoseTrack](https://posetrack.net/)) to test it. 

- We'll need to change the settings of these 2 nodes, in order to set the input and output directories, as follows:
  ```
  nodes:
    - input.recorded:   # note the ":"
      - input_dir: <directory where videos/images are stored>
    - model.posenet
    - draw.poses
    - output.media_writer:  # note the ":"
      - output_dir: <directory to save results>
  ```
- Once PeekingDuck has finished running, the processed files will be saved to the specified output directory. If you've used the short sample video, open the processed file and you should get this:

  <img src="images/readme/posenet_running.gif" width="50%">

- To find out what other settings can be tweaked for different nodes, check out PeekingDuck's [node glossary](docs/node_glossary.md).

## Explore PeekingDuck Nodes

AI models are cool and fun, but we're even more interested to use them to solve real-world problems. We've combined heuristic nodes with model nodes to create **use cases**, such as [social distancing](https://aisingapore.org/2020/06/hp-social-distancing/) and [group size checking](https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker/) to help combat COVID-19. For more details, click on the heading of each use case below.

| | |
|-|-|
| [Social Distancing](docs/use_cases/social_distancing.md) | [Zone Counting](docs/use_cases/zone_counting.md) |
|<img src="images/readme/social_distancing.gif" width="100%"> |<img src="images/readme/zone_counting.gif" width="100%">|
| [Group Size Checking](docs/use_cases/group_size_checking.md) | [Object Counting](docs/use_cases/object_counting.md) |
|<img src="images/readme/group_size_check_2.gif" width="100%">|<img src="images/readme/object_counting.gif" width="100%"> |
| | |

We're constantly developing new nodes to increase PeekingDuck's capabilities. You've gotten a taste of some of our commonly used nodes in the previous demos, but PeekingDuck can do a lot more. To see what other nodes available, check out the [node glossary](docs/node_glossary.md).

## Create Custom Nodes

You may need to create your own custom nodes. Perhaps you'd like to take a snapshot of a video frame, and post it to your API endpoint; perhaps you have a model trained on a custom dataset, and would like to use PeekingDuck's `input`, `draw`, and `output` nodes. 

We've designed PeekingDuck to be very flexible - you can create your own nodes and use them with ours. This [guide](docs/guide_custom_nodes.md) provides more details on how to do that.

## Acknowledgements

This project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG-RP-2019-050). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

## License

PeekingDuck is under the open source [Apache License 2.0](LICENSE) (:

Even so, your organisation may require legal proof of its right to use PeekingDuck, due to circumstances such as the following:
- Your organisation is using PeekingDuck in a jurisdiction that does not recognise this license
- Your legal department requires a license to be purchased
- Your organisation wants to hold a tangible legal document as evidence of the legal right to use and distribute PeekingDuck

[Contact us](https://aisingapore.org/home/contact/) if any of these circumstances apply to you.


