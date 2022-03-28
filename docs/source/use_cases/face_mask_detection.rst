*******************
Face Mask Detection
*******************

Overview
========

Wearing of face masks in public places can help prevent the spread of COVID-19 and other infectious
diseases. AI Singapore has developed a solution that checks whether or not a person is wearing a
face mask. This can be used in places such as malls or shops to ensure that visitors adhere to
the guidelines.

.. image:: /assets/use_cases/face_mask_detection.gif
   :class: no-scaled-link
   :width: 50 %

We have trained a custom YOLOv4 model to detect whether or not a person is wearing a face mask.
This is explained in the `How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: face_mask_detection.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/face_mask_detection.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

The main component is the detection of face mask using the custom YOLOv4 model.

**Face Mask Detection**

We use an open source object detection model known as `YOLOv4 <https://arxiv.org/abs/2004.10934>`_
and its smaller and faster variant known as YOLOv4-tiny to identify the bounding boxes of human
faces with and without face masks. This allows the application to identify the locations of faces
and their corresponding classes (no_mask = 0 or mask = 1) in a video feed. Each of these locations
are represented as a pair of `x, y` coordinates in the form :math:`[x_1, y_1, x_2, y_2]`, where
:math:`(x_1, y_1)` is the top-left corner of the bounding box, and :math:`(x_2, y_2)` is the bottom
right. These are used to form the bounding box of each human face detected.

The :mod:`model.yolo_face` node detects human faces with and without face masks using the
YOLOv4-tiny model by default. The classes are differentiated by the labels and the colors of the
bounding boxes when multiple faces are detected. For more information on how to adjust the
``yolo_face`` node, check out its :doc:`configurable parameters </nodes/model.yolo_face>`.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.visual:
       source: 0
   - model.yolo_face
   - draw.bbox:
       show_labels: true
   - output.screen

**1. Face Mask Detection Node**

The :mod:`model.yolo_face` node is used for face detection and to classify if the face is masked or
unmasked. To simply detect faces without needing to classify if the face is masked, you can also
consider the :mod:`model.mtcnn` node.

**2. Adjusting Nodes**

Some common node behaviors that you might want to adjust are:

* ``model_type``: This specifies the variant of YOLOv4 to be used. By default, the `v4tiny` model
  is used, but for better accuracy, you may want to try the `v4` model.
* ``detect_ids``: This specifies the class to be detected where no_mask = 0 and mask = 1. By default,
  the model detects faces with and without face masks (default = [0, 1]).
* ``score_threshold``: This specifies the threshold value. Bounding boxes with confidence score
  lower than the threshold are discarded. You may want to lower the threshold value to increase the
  number of detections.
