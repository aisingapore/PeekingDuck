<div align="center">
    <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/peekingduck.png" width="30%">
    <h1>PeekingDuck</h1>
</div>

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)](https://pypi.org/project/peekingduck/)
[![PyPI version](https://badge.fury.io/py/peekingduck.svg)](https://pypi.org/project/peekingduck/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/peekingduck)](https://pypi.org/project/peekingduck/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/peekingduck/badge/?version=stable)](https://peekingduck.readthedocs.io/en/stable/?badge=stable)

## What is PeekingDuck?

PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV) inference. It helps to significantly cut down development time when building CV pipelines. The name "PeekingDuck" is a play on these words: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing).


## Install and Run PeekingDuck

1. Install PeekingDuck from [PyPI](https://pypi.org/project/peekingduck/).
    ```
    > pip install peekingduck
    ```
    *Note: if installing on a ARM-based device such as a Raspberry Pi or M1 Macbook, include the `--no-dependencies` flag, and separately install other dependencies listed in our [requirements.txt](https://github.com/aimakerspace/PeekingDuck/blob/dev/requirements.txt). See our guide for [M1 Mac](https://peekingduck.readthedocs.io/en/stable/getting_started/01_installation.html#m1-mac-installation) installation.*

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
              └── configs
    ```

3. Run a demo.
    ```
    > peekingduck run
    ```

    If you have a webcam, you should see the demo running live:

    <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/yolo_demo.gif" width="50%">

    The previous command looks for a `run_config.yml` in the current directory. You can also specify the path of a different config file to be used, as follows:
    ```
    > peekingduck run --config_path <path_to_config>
    ```

    Terminate the program by clicking on the output screen and pressing `q`.

4. For more help on how to use PeekingDuck's command line interface, you can use `peekingduck --help`.


## How PeekingDuck Works

**Nodes** are the building blocks of PeekingDuck. Each node is a wrapper for a Python function, and contains information on how other PeekingDuck nodes may interact with it.

PeekingDuck has 6 types of nodes:

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/diagrams/node_types.drawio.svg">

A **pipeline** governs the behavior of a chain of nodes. The diagram below shows the pipeline used in the previous demo. Nodes in a pipeline are called in sequential order, and the output of one node will be the input to another. For example, `input.live` produces "img", which is taken in by `model.yolo`, and `model.yolo` produces "bboxes", which is taken in by `draw.bbox`. For ease of visualisation, not all the inputs and outputs of these nodes are included in this diagram.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/diagrams/yolo_demo.drawio.svg">

To list the available nodes in PeekingDuck and get their respective documentation's URL: 
 ```
 > peekingduck nodes
 ```


## Explore PeekingDuck's Features

You can find the complete documentation for PeekingDuck at our [Read the Docs site](https://peekingduck.readthedocs.io/en/stable/). This includes information on:
- [Changing PeekingDuck nodes](https://peekingduck.readthedocs.io/en/stable/getting_started/02_configure_pkdk.html) and their settings
- [Official documentation](https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.html) for all PeekingDuck nodes, describing their behaviour, inputs, outputs and settings
- Creating your own [custom nodes](https://peekingduck.readthedocs.io/en/stable/getting_started/03_custom_nodes.html), and using them with PeekingDuck nodes
- Using PeekingDuck as an [imported Python module](https://peekingduck.readthedocs.io/en/stable/getting_started/04_import_peekingduck.html) within your project
- Benchmarks and class/keypoints IDs for [object detection](https://peekingduck.readthedocs.io/en/stable/resources/01a_object_detection.html) and [pose estimation](https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html) models.

You are also welcome to join discussions about using PeekingDuck in the following channels:
- [Github discussion board](https://github.com/aimakerspace/PeekingDuck/discussions)
- [AI Singapore's Community forum](https://community.aisingapore.org/groups/computer-vision/forum/)


## PeekingDuck Use Cases

AI models are cool and fun, but we're even more interested to use them to solve real-world problems. We've combined dabble nodes with model nodes to create **use cases**, such as [social distancing](https://aisingapore.org/2020/06/hp-social-distancing/) and [group size checking](https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker/) to help combat COVID-19. For more details, click on the heading of each use case below.

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Social Distancing](https://peekingduck.readthedocs.io/en/stable/use_cases/social_distancing.html) | [Zone Counting](https://peekingduck.readthedocs.io/en/stable/use_cases/zone_counting.html) |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/social_distancing.gif" width="100%"> | <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/zone_counting.gif" width="100%"> |
| [Group Size Checking](https://peekingduck.readthedocs.io/en/stable/use_cases/group_size_checking.html) | [Object Counting](https://peekingduck.readthedocs.io/en/stable/use_cases/object_counting.html) |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/group_size_check_2.gif" width="100%"> | <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/object_counting.gif" width="100%"> |
| [Privacy Protection (Faces)](https://peekingduck.readthedocs.io/en/stable/use_cases/privacy_protection_faces.html) | [Privacy Protection (License Plates)](https://peekingduck.readthedocs.io/en/stable/use_cases/privacy_protection_license_plate.html) |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/privacy_protection_faces.gif" width="100%"> | <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/privacy_protection_license_plates.gif" width="100%"> |
| [Face Mask Detection](https://peekingduck.readthedocs.io/en/stable/use_cases/face_mask_detection.html) |                                                              |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/mask_detection.gif" width="100%"> |                                                              |

We're constantly developing new nodes to increase PeekingDuck's capabilities. You've gotten a taste of some of our commonly used nodes in the previous demos, but PeekingDuck can do a lot more. To see what other nodes are available, check out PeekingDuck's [API Reference](https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.html).


## Acknowledgements

This project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG-RP-2019-050). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

## License

PeekingDuck is under the open source [Apache License 2.0](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE) (:

Even so, your organisation may require legal proof of its right to use PeekingDuck, due to circumstances such as the following:
- Your organisation is using PeekingDuck in a jurisdiction that does not recognise this license
- Your legal department requires a license to be purchased
- Your organisation wants to hold a tangible legal document as evidence of the legal right to use and distribute PeekingDuck

[Contact us](https://aisingapore.org/home/contact/) if any of these circumstances apply to you.

## Additional References
Additional references can be found [here](https://peekingduck.readthedocs.io/en/stable/resources/02_bibliography.html).
