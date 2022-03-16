*******************
Face Mask Detection
*******************

Overview
========

As part of COVID-19 measures, the Singapore Government has mandated the wearing of face masks in
public places. AI Singapore has developed a solution that checks whether or not a person is wearing
a face mask. This can be used in places such as in malls or shops to ensure that visitors adhere to
the guidelines.

.. image:: /assets/use_cases/face_mask_detection.gif
   :class: no-scaled-link
   :width: 70 %

We have trained a custom YOLOv4 model to detect whether or not a person is wearing a face mask.
This is explained in the `How it Works`_ section.

Demo
====

.. |pipeline_config| replace:: face_mask_detection.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/face_mask_detection.yml

To try our solution on your own computer, :doc:`install </getting_started/02_basic_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |pipeline_config|\ >

How it Works
============

The main component is the detection of face mask using the custom YOLOv4 model.

**Face Mask Detection**

We use an open source object detection model known as `YOLOv4 <https://arxiv.org/abs/2004.10934>`_
and its smaller and faster variant known as YOLOv4-tiny to identify the bounding boxes of human
faces with and without face masks. This allows the application to identify the locations of faces
and their corresponding classes (no_mask = 0 or mask = 1) in a video feed. Each of these locations
are represented as a pair of `x, y` coordinates in the form :math:`[x_1, y_1, x_2, y_2]`, where
:math:`(x_1, y_1)` is the top left corner of the bounding box, and :math:`(x_2, y_2)` is the bottom
right. These are used to form the bounding box of each human face detected.

The :mod:`model.yolo_face` node detects human faces with and without face masks using the
YOLOv4-tiny model by default. The classes are differentiated by the labels and the colors of the
bounding boxes when multiple faces are detected. For more information on how adjust the
``yolo_face`` node, check out its :doc:`configurable parameters </nodes/model.yolo_face>`.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.live
   - model.yolo_face
   - dabble.fps
   - draw.bbox:
       show_labels: true
   - draw.legend
   - output.screen

**1. Face Mask Detection Node**

By default, the node uses the YOLOv4-tiny model for face detection. For better accuracy, you can
try the :mod:`YOLOv4 model <model.yolo_face>` that is included in our repo.

**2. Adjusting Nodes**

Some common node behaviors that you might want to adjust are:

* `detect_ids`: This specifies the class to be detected where no_mask = 0 and mask = 1. By default,
  the model detects faces with and without face masks (default = [0, 1]).
* `yolo_score_threshold`: This specifies the threshold value. Bounding boxes with confidence score
  lower than the threshold are discarded. You may want to lower the threshold value to increase the
  number of detections.
