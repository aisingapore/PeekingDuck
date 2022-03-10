***********************************
Privacy Protection (License Plates)
***********************************

Overview
========

Posting images or videos of our vehicles online might lead to others misusing our license plate
number to reveal our personal information or being vulnerable to license plate cloning. Hence, AI
Singapore has developed a solution that performs license plate anonymization. This can also be used
to comply with the General Data Protection Regulation (GDPR) or other data privacy laws.

.. image:: /assets/use_cases/privacy_protection_license_plates.gif
   :class: no-scaled-link
   :width: 100 %

Our solution automatically detects and blurs vehicles' license plates. This is explained in the `How it Works`_ section.

Demo
====

.. |pipeline_config| replace:: privacy_protection_license_plates.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_protection_license_plates.yml

To try our solution on your own computer, :doc:`install </getting_started/02_basic_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |pipeline_config|\ >

How it Works
============

There are two main components to license plate anonymization:

#. License plate detection using AI and
#. License plate de-identification.

**1. License Plate Detection**

We use open-source object detection models under the `YOLOv4 <https://arxiv.org/abs/2004.10934>`_
family to identify the locations of the license plates in an image/video feed. Specifically, we
offer the YOLOv4-tiny model, which is faster, and the YOLOv4 model, which provides higher accuracy.
The locations of detected license plates are returned as an array of coordinates in the form
:math:`[x_1, y_1, x_2, y_2]`, where :math:`(x_1, y_1)` is the top left corner of the bounding box,
and :math:`(x_2, y_2)` is the bottom right. These are used to form the bounding box of each license
plate detected. For more information on how to adjust the license plate detector node, check out
the :doc:`license plate detector configurable parameters </nodes/model.yolo_license_plate>`.

**2. License Plate De-Identification**

To perform license plate de-identification, the areas bounded by the bounding boxes are blurred
using a Gaussian function (Gaussian blur).

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.recorded:
       input_dir: <path/to/video with cars>
   - model.yolo_license_plate
   - dabble.fps
   - draw.blur_bbox
   - draw.legend
   - output.screen
   
**1. License Plate Detection Node**

By default, the license plate detection node uses the YOLOv4 model to detect license plates. When
faster inference speed is required, you can change the parameter in the run config declaration to
use the YOLOv4-tiny model:

.. code-block:: yaml

   - model.yolo_license_plate:
       model_type: v4tiny

**2. License Plate De-Identification Nodes**

You can choose to mosaic or blur the detected license plate using the :mod:`draw.mosaic_bbox` or
:mod:`draw.blur_bbox` node in the run config declaration.

**3. Adjusting Nodes**

With regard to the YOLOv4 model, some common node configurations that you might want to adjust are:

* ``yolo_score_threshold``: The bounding boxes with confidence score less than the specified score
  threshold are discarded. (default = 0.1)
* ``yolo_iou_threshold``: The overlapping bounding boxes above the specified Intersection over
  Union (IoU) threshold are discarded. (default = 0.3)

In addition, some common node behaviors that you might want to adjust for the
:mod:`dabble.mosaic_bbox` and :mod:`dabble.blur_bbox` nodes are:

* ``mosaic_level``: Defines the resolution of a mosaic filter (:math:`width \times height`); the
  value corresponds to the number of rows and columns used to create a mosaic. (default = 7) For
  example, the default value creates a :math:`7 \times 7` mosaic filter. Increasing the number
  increases the intensity of pixelation over an area.
* ``blur_level``:  Defines the standard deviation of the Gaussian kernel used in the Gaussian
  filter. (default = 50) The higher the blur level, the more intense is the blurring.
