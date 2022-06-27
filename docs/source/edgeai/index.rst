:orphan:

.. _edge_ai:

*******
Edge AI
*******

PeekingDuck supports running optimized TensorRT models on the Nvidia Jetson family of
devices for Edge AI.
Using the TensorRT model on a Jetson device provides a speed boost over the regular
Tensorflow/PyTorch version.

Currently, PeekingDuck includes TensorRT versions of the following models:

#. Movenet model for pose estimation,
#. YoloX model for object detection.


Using TensorRT Models
=====================

To use the TensorRT version of a model, change the ``model_format`` of the model
configuration to ``tensorrt``.

The following ``pipeline_config.yml`` shows how to use the Movenet TensorRT model 
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


The following ``pipeline_config.yml`` shows how to use the YoloX TensorRT model 
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


Performace Speedup
==================

The following charts show the speed up obtainable with the TensorRT models.
The numbers were obtained from our in-house testing with the actual devices.


Nvidia Jetson Xavier NX with 8GB RAM
------------------------------------

.. figure:: /assets/charts/tensorrt_nx_movenet_fps.png
.. figure:: /assets/charts/tensorrt_nx_yolox_fps.png


Nvidia Jetson Xavier AGX with 16GB RAM
--------------------------------------

This Jetson Xavier AGX device was loaned to us by 
`Advantech Singapore <https://www.advantech.com>`__.

.. figure:: /assets/charts/tensorrt_agx_movenet_fps.png
.. figure:: /assets/charts/tensorrt_agx_yolox_fps.png

