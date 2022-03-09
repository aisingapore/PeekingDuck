****************************
Human Detection and Tracking
****************************

Overview
========

Multiple Object Tracking (MOT) aims at estimating bounding boxes and identities of objects in
videos. AI Singapore has developed a solution that performs human detection and tracking in a
single model. This application can have a wide range of applications, starting from video
surveillance and human computer interaction to robotics.

.. image:: /assets/use_cases/human_tracking.gif
   :class: no-scaled-link
   :width: 100 %

Our solution automatically detects and tracks human persons. This is explained in the `How it Works`_ section.

Demo
====

.. |pipeline_config| replace:: human_tracking.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/human_tracking.yml

To try our solution on your own computer, :doc:`install </getting_started/02_basic_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |pipeline_config|\ >

How it Works
============

There are two main components to this MOT:

#. Human target detection using AI and
#. Corresponding appearance embedding. 

**1. Human Detection**

We use an open source detection model trained on pedestrian detection and person search datasets
known as `JDE <https://arxiv.org/abs/1909.12605>`_ to identify human persons. This allows the
application to identify the locations of human persons in a video feed. Each of these locations is
represented as a pair of `x, y` coordinates in the form :math:`[x_1, y_1, x_2, y_2]`, where
:math:`(x_1, y_1)` is the top left corner of the bounding box, and :math:`(x_2, y_2)` is the bottom
right. These are used to form the bounding box of each human person detected. For more information
on how adjust the JDE node, check out the :doc:`JDE configurable parameters </nodes/model.jde>`.

**2. Appearance Embedding Tracking**

To perform tracking, JDE models the training process as a multi-task learning problem with anchor
classification, box regression, and embedding learning. The model outputs a ``track_id`` for each
detection based on the appearance embedding learned.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.live
   - model.jde
   - dabble.fps
   - draw.bbox
   - draw.tag
   - draw.legend
   - output.screen

**1. JDE Node**

This node employs a single network to **simultaneously** output detection results and the
corresponding appearance embeddings of the detected boxes. Therefore JDE stands for Joint Detection
and Embedding. Check out the :doc:`node documentation </nodes/model.jde>` for more information
regarding the model, i.e., research paper and repository.

JDE employs a DarkNet-53 `YOLOv3 <https://arxiv.org/abs/1804.02767>`_ as the backbone network for
human detection. To learn appearance embeddings, a metric learning algorithm with triplet loss
together is used. Observations are assigned to tracklets using the Hungarian algorithm. The Kalman
filter is used to smooth the trajectories and predict the locations of previous tracklets in the
current frame.

**2. Adjusting Node**

With regard to the JDE model node, some common behaviors that you might want to adjust are:

* ``iou_threshold``: Specifies the threshold value for Intersection over Union of detections
  (default = 0.5). 
* ``score_threshold``: Specifies the threshold values for the detection confidence (default = 0.5).
  You may want to lower this value to increase the number of detections.
* ``nms_threshold``: Specifies the threshold value for non-maximal suppression (default = 0.4).
  You may want to lower this value to increase the number of detections.
* ``min_box_area``: Minimum value for area of detected bounding box. Calculated by :math:`width \times height`.
* ``track_buffer``: Specifies the threshold to remove track if track is lost for more
  frames than this value.
