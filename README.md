<!-- <div align="center">
    <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/docs/source/assets/peekingduck.png" width="30%">
    <h1>PeekingDuck</h1>
</div> -->
<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/peekingduck.png" width="26%" align="right" style="padding: 10px 0px 10px 10px;">

PeekingDuck
===========

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://pypi.org/project/peekingduck/)
[![PyPI version](https://badge.fury.io/py/peekingduck.svg)](https://pypi.org/project/peekingduck/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/peekingduck)](https://pypi.org/project/peekingduck/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/peekingduck/badge/?version=stable)](https://peekingduck.readthedocs.io/en/stable/?badge=stable)

**PeekingDuck** is an open-source, modular framework in Python, built for Computer Vision (CV) inference. It helps to significantly cut down development time when building CV pipelines. The name "PeekingDuck" is a play on these words: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing).


Installation
------------

Install from [PyPI](https://pypi.org/project/peekingduck/) using `pip`

```
> pip install peekingduck
```

*Note: for users with ARM-based devices such as a Raspberry Pi or M1 MacBook, please refer to the documentation for more detailed [installation instructions](https://peekingduck.readthedocs.io/en/docs-v1.2/getting_started/03_advanced_install.html).*

PeekingDuck can also be [installed in a virtual environment](https://peekingduck.readthedocs.io/en/docs-v1.2/getting_started/02_basic_install.html).


Usage
-----

Create a project folder and initialize a PeekingDuck project.
```
> mkdir <project_dir>
> cd <project_dir>
> peekingduck init
```

Run the demo pipeline.
```
> peekingduck run
```

If you have a webcam, you should see the demo running live:

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/yolo_demo.gif" width="50%">

Terminate the program by clicking on the output screen and pressing `q`.

Use `peekingduck --help` to display help information for PeekingDuck's command line interface.


Documentation
-------------

The [complete documentation](https://peekingduck.readthedocs.io/en/docs-v1.2/) for PeekingDuck includes information on:
- [Tutorials](https://peekingduck.readthedocs.io/en/docs-v1.2/tutorials/index.html) to get you familiarized with PeekingDuck's features
- [API documentation](https://peekingduck.readthedocs.io/en/docs-v1.2/master.html#api-documentation) for all PeekingDuck nodes and their configurations
- [Benchmarks](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/index.html) for various models that comes with PeekingDuck.
- [Class IDs](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/01a_object_detection.html#object-detection-ids) for object detection models
- [Keypoint IDs](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/01b_pose_estimation.html#keypoint-ids) for pose estimation models
- [Use case solutions](https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/index.html) created using PeekingDuck nodes


Acknowledgements
----------------

This project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG-RP-2019-050). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.


License
-------

PeekingDuck is under the open source [Apache License 2.0](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE) (:

Even so, your organization may require legal proof of its right to use PeekingDuck, due to circumstances such as the following:
- Your organization is using PeekingDuck in a jurisdiction that does not recognize this license
- Your legal department requires a license to be purchased
- Your organization wants to hold a tangible legal document as evidence of the legal right to use and distribute PeekingDuck

[Contact us](https://aisingapore.org/home/contact/) if any of these circumstances apply to you.


Communication
-------------

- [Github discussion board](https://github.com/aimakerspace/PeekingDuck/discussions)
- [AI Singapore's Community forum](https://community.aisingapore.org/groups/computer-vision/forum/)