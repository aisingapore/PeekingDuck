:orphan:

.. include:: /include/substitution.rst

.. _edge_ai:

*******
Edge AI
*******

PeekingDuck supports running optimized TensorRT [1]_ models on devices with NVIDIA GPUs.
Using the TensorRT model on these devices provides a speed boost over the regular
TensorFlow/PyTorch version.
A potential use case is running PeekingDuck on an NVIDIA Jetson device for Edge AI
inference.

Currently, PeekingDuck includes TensorRT versions of the following models:

#. MoveNet model for pose estimation,
#. YOLOX model for object detection.


Using TensorRT Models
=====================

To use the TensorRT version of a model, change the ``model_format`` of the model
configuration to ``tensorrt``.

The following ``pipeline_config.yml`` shows how to use the MoveNet TensorRT model 
for pose estimation:

.. code-block:: yaml
    :linenos:

    nodes:
    - input.visual:
        source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
    - model.movenet:
        model_format: tensorrt
        model_type: singlepose_lightning
    - draw.poses
    - dabble.fps
    - draw.legend:
        show: ["fps"]
    - output.screen


The following ``pipeline_config.yml`` shows how to use the YOLOX TensorRT model 
for object detection:

.. code-block:: yaml
    :linenos:

    nodes:
    - input.visual:
        source: https://storage.googleapis.com/peekingduck/videos/cat_and_computer.mp4
    - model.yolox:
        detect: ["cup", "cat", "laptop", "keyboard", "mouse"]
        model_format: tensorrt
        model_type: yolox-tiny
    - draw.bbox:
        show_labels: True    # configure draw.bbox to display object labels
    - dabble.fps
    - draw.legend:
        show: ["fps"]
    - output.screen


Performance Speedup
===================

The following charts show the speed up obtainable with the TensorRT models.
The numbers were obtained from our in-house testing with the actual devices.


NVIDIA Jetson Xavier NX with 8GB RAM
------------------------------------

.. figure:: /assets/charts/tensorrt_nx_movenet_fps.png
.. figure:: /assets/charts/tensorrt_nx_yolox_fps.png

    Jetson Xavier NX specs used for testing: [2]_

    ``CPU``: 6 cores (6MB L2 + 4MB L3) |br|
    ``GPU``: 384-core Volta, 48 Tensor cores |br|
    ``RAM``: 8 GB


NVIDIA Jetson Xavier AGX with 16GB RAM
--------------------------------------

.. figure:: /assets/charts/tensorrt_agx_movenet_fps.png
.. figure:: /assets/charts/tensorrt_agx_yolox_fps.png

    Jetson Xavier AGX specs used for testing: [3]_

    ``CPU``: 8 cores (8MB L2 + 4MB L3) |br|
    ``GPU``: 512-core Volta, 64 Tensor cores |br|
    ``RAM``: 16 GB


Test Conditions
^^^^^^^^^^^^^^^

The following test conditions were followed:
 | - :mod:`input.visual`, the model of interest, and :mod:`dabble.fps` nodes were used to perform
     inference on videos
 | - 2 videos were used to benchmark each model, one with only 1 human (``single``), and the other
     with multiple humans (``multiple``)
 | - Both videos are about 1 minute each, recorded at ~30 FPS, which translates to about 1,800
     frames to process per video
 | - 1280×720 (HD ready) resolution was used, as a bridge between 640×480 (VGA) of poorer quality
     webcams, and 1920×1080 (Full HD) of CCTVs
 | - FPS numbers are averaged over 5 separate runs


References
==========

.. [1] `NVIDIA TensorRT Reference <https://developer.nvidia.com/tensorrt>`_
.. [2] `NVIDIA Jetson Xavier NX Tech Specs <https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit>`_
.. [3] `NVIDIA Jetson Xavier AGX Tech Specs <https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit>`_