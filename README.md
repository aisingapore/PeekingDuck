<br />

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/peekingduck.svg">

---

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://pypi.org/project/peekingduck/)
[![PyPI version](https://badge.fury.io/py/peekingduck.svg)](https://pypi.org/project/peekingduck/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/peekingduck)](https://pypi.org/project/peekingduck/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/peekingduck/badge/?version=stable)](https://peekingduck.readthedocs.io/en/stable/?badge=stable)

<h4 align="center">
  <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/getting_started/index.html">Getting started</a>
  <span> · </span>
  <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/tutorials/index.html">Tutorials</a>
  <span> · </span>
  <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/master.html#api-documentation">API documentation</a>
  <span> · </span>
  <a href="https://github.com/aimakerspace/PeekingDuck/issues">Report a bug</a>
  <span> · </span>
  <a href="#communities">Communities</a>
</h4>

---

**PeekingDuck** is an open-source, modular framework in Python, built for Computer Vision (CV) inference. The name "PeekingDuck" is a play on: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing).


Features
--------

### Build realtime CV pipelines
* PeekingDuck enables you to build powerful CV pipelines with minimal lines of code.

### Leverage on various SOTA models
* PeekingDuck comes with various [object detection](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/01a_object_detection.html), [pose estimation](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/01b_pose_estimation.html), [object tracking](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/01c_object_tracking.html), and [crowd counting](https://peekingduck.readthedocs.io/en/docs-v1.2/resources/01d_crowd_counting.html) models. Mix and match different nodes to construct solutions for various [use cases](https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/index.html).

### Create custom nodes
* You can create [custom nodes](https://peekingduck.readthedocs.io/en/docs-v1.2/tutorials/02_duck_confit.html#custom-nodes) to meet your own project's requirements. PeekingDuck can also be [imported as a library](https://peekingduck.readthedocs.io/en/docs-v1.2/tutorials/04_import_as_module.html) to fit into your existing workflows.


Installation
------------

Install from [PyPI](https://pypi.org/project/peekingduck/) using `pip`

```
> pip install peekingduck
```

*Note: for users with ARM-based devices such as a Raspberry Pi or M1 Mac, please refer to the documentation for more detailed [installation instructions](https://peekingduck.readthedocs.io/en/docs-v1.2/getting_started/03_advanced_install.html).*

PeekingDuck can also be [installed in a virtual environment](https://peekingduck.readthedocs.io/en/docs-v1.2/getting_started/02_basic_install.html).


```
> peekingduck --verify_install
```

You should see a video of a person waving his hand with
[bounding boxes overlaid](https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/getting_started/verify_install.gif).

The video will auto-close when it is run to the end, select the video window and press `q` to exit earlier.


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

If you have a webcam, you should see yourself on the output screen with
[skeletal frame overlaid](https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/posenet_demo.gif).

Terminate the program by clicking on the output screen and pressing `q`.

Use `peekingduck --help` to display help information for PeekingDuck's command line interface.


Gallery
-------

<table>
  <tr>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/social_distancing.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/social_distancing.gif">
      </a>
    </td>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/privacy_protection_faces.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/privacy_protection_faces.gif">
      </a>
    </td>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/zone_counting.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/zone_counting.gif">
      </a>
    </td>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/multiple_object_tracking.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/multiple_object_tracking.gif">
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/group_size_checking.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/group_size_checking.gif">
      </a>
    </td>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/privacy_protection_license_plates.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/privacy_protection_license_plates.gif">
      </a>
    </td>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/crowd_counting.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/crowd_counting.gif">
      </a>
    </td>
    <td>
      <a href="https://peekingduck.readthedocs.io/en/docs-v1.2/use_cases/human_tracking.html">
        <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/docs-v1.2/docs/source/assets/use_cases/human_tracking.gif">
      </a>
    </td>
  </tr>
</table>


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


Communities
-----------

- [AI Singapore community forum](https://community.aisingapore.org/groups/computer-vision/forum/)
- [Discuss on GitHub](https://github.com/aimakerspace/PeekingDuck/discussions)
