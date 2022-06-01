***********************************
Privacy Protection (License Plates)
***********************************

Overview
========

Posting images or videos of our vehicles online might lead to others misusing our license plate
numbers to reveal our personal information. Our solution performs license plate anonymization,
and can also be used to comply with the General Data Protection Regulation (GDPR) or other data
privacy laws.

.. image:: /assets/use_cases/privacy_protection_license_plates.gif
   :class: no-scaled-link
   :width: 50 %

Our solution automatically detects and blurs vehicles' license plates. This is explained in the
`How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: privacy_protection_license_plates.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/main/use_cases/privacy_protection_license_plates.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

There are two main components to license plate anonymization:

#. License plate detection, and
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
using a Gaussian blur function.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.visual:
       source: <path/to/video with cars>
   - model.yolo_license_plate
   - draw.blur_bbox
   - output.screen
   
**1. License Plate Detection Node**

By default, :mod:`model.yolo_license_plate` uses the ``v4`` model type to detect license plates.
If faster inference speed is required, the ``v4tiny`` model type can be used instead. 

**2. License Plate De-Identification Nodes**

You can choose to mosaic or blur the detected license plate using the :mod:`draw.mosaic_bbox` or
:mod:`draw.blur_bbox` node in the run config declaration.

.. figure:: /assets/use_cases/privacy_protection_license_plates_comparison.jpg
   :alt: De-identification effect comparison
   :class: no-scaled-link
   :width: 50 %

   De-identification with mosaic (left) and blur (right).

**3. Adjusting Nodes**

With regard to the YOLOv4 model, some common node configurations that you might want to adjust are:

* ``score_threshold``: The bounding boxes with confidence score less than the specified score
  threshold are discarded. (default = 0.1)
* ``iou_threshold``: The overlapping bounding boxes above the specified Intersection over
  Union (IoU) threshold are discarded. (default = 0.3)

In addition, some common node behaviors that you might want to adjust for the
:mod:`dabble.mosaic_bbox` and :mod:`dabble.blur_bbox` nodes are:

* ``mosaic_level``: Defines the resolution of a mosaic filter (:math:`width \times height`); the
  value corresponds to the number of rows and columns used to create a mosaic. (default = 7) For
  example, the default value creates a :math:`7 \times 7` mosaic filter. Increasing the number
  increases the intensity of pixelization over an area.
* ``blur_level``:  Defines the standard deviation of the Gaussian kernel used in the Gaussian
  filter. (default = 50) The higher the blur level, the greater the blur intensity.
