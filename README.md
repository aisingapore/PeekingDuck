<div align="center">
    <img src="images/peekingduck.JPG" width="30%">
    <h1>PeekingDuck</h1>
</div>

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)](https://pypi.org/project/peekingduck/)
[![PyPI version](https://badge.fury.io/py/peekingduck.svg)](https://pypi.org/project/peekingduck/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## What is PeekingDuck?

PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV) inference. It helps to significantly cut down development time when building CV pipelines. The name "PeekingDuck" is a play on these words: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing).

## Install PeekingDuck

1. Install PeekingDuck from [PyPI](https://pypi.org/project/peekingduck/).
    ```
    pip install peekingduck
    ```

2. Create a project folder at a convenient location.
    ```
    mkdir <project_name>
    cd <project_name>
    ```

3. Pull default configurations and run a demo.
    ```
    peekingduck get-configs
    peekingduck run
    ```

    If you have a webcam, you should be able to see the demo running live.

    <img src="images/black.png" width="50%">

## How PeekingDuck Works

Nodes are the basic blocks of PeekingDuck. A node is a wrapper for a Python function, and contains information on how other PeekingDuck nodes may interact with it. 

PeekingDuck has 5 types of nodes:
- input: responsible for dealing with various types of inputs such as live video feeds, recorded videos/images, etc, and passing individual frames to other nodes
- model: AI models for object detection, pose estimation, etc
- heuristic: broad class of functions such as algorithms or approximations that transform model results into useful outputs
- draw: draw results such as bounding boxes on frames
- output: responsible for showing results on screen, saving output videos, posting to API endpoint etc

A pipeline governs the behavior of a chain of nodes. Nodes in a pipeline are called in sequential order, and the output of one node will be the input to another. The diagram below shows the pipeline used in the above demo.


<img src="diagrams/yolo_demo.drawio.svg">