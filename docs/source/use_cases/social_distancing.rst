*****************
Social Distancing
*****************

Overview
========

To support the fight against COVID-19, AI Singapore developed a solution to encourage individuals
to maintain physical distance from each other. This can be used in many places, such as in malls to
encourage social distancing in long queues, or in workplaces to ensure employees' well-being. An
example of the latter is `HP Inc. <https://aisingapore.org/2020/06/hp-social-distancing>`_, which
collaborated with us to deploy this solution on edge devices in its manufacturing facility in
Singapore.

.. image:: /assets/use_cases/social_distancing.gif
   :class: no-scaled-link
   :width: 50 %

The most accurate way to measure distance is to use a 3D sensor with depth perception, such as a
RGB-D camera or a `LiDAR <https://en.wikipedia.org/wiki/Lidar>`_. However, most cameras such as
CCTVs and IP cameras usually only produce 2D videos. We developed heuristics that are able to give
an approximate measure of physical distance from 2D videos, addressing this limitation. This is
explained in the `How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: social_distancing.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/social_distancing.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

There are two main components to obtain the distance between individuals:
#. Human pose estimation using AI, and
#. Depth and distance approximation using heuristics.

**1. Human Pose Estimation**

We use an open source human pose estimation model known as `PoseNet <https://arxiv.org/abs/1505.07427>`_
to identify key human skeletal points. This allows the application to identify where individuals
are located within the video feed. The coordinates of the various skeletal points will then be used
to determine the distance between individuals.

.. image:: /assets/use_cases/posenet_demo.gif
   :class: no-scaled-link
   :width: 50 %

**2. Depth and Distance Approximation**

To measure the distance between individuals, we have to estimate the 3D world coordinates from the
keypoints in 2D coordinates. To achieve this, we compute the depth :math:`Z` from the `x, y` coordinates
using the relationship below:

.. image:: /assets/use_cases/distance_estimation.png
   :class: no-scaled-link
   :width: 50 %

where:

* :math:`Z` = depth or distance of scene point from camera
* :math:`f` = focal length of camera
* :math:`y` = `y` position of image point
* :math:`Y` = `y` position of scene point

:math:`Y_1 - Y_2` is a reference or "ground truth length" that is required to obtain the depth.
After numerous experiments, it was decided that the optimal reference length would be the average
height of a human torso (height from human hip to center of face). Width was not used as this value
has high variance due to the different body angles of an individual while facing the camera.

Once we have the 3D world coordinates of the individuals in the video, we can compare the distances
between each pair of individuals and check if they are too close to each other.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.visual:
       source: 0
   - model.posenet
   - dabble.keypoints_to_3d_loc:
       focal_length: 1.14
       torso_factor: 0.9
   - dabble.check_nearby_objs:
       near_threshold: 1.5
       tag_msg: "TOO CLOSE!"
   - draw.poses
   - draw.tag:
       show: ["flags"]
   - output.screen

**1. Pose Estimation Model**

By default, we are using the PoseNet model with a ResNet backbone for pose estimation. Please take
a look at the :doc:`benchmarks </resources/01b_pose_estimation>` of pose estimation models that
are included in PeekingDuck if you would like to use a different model or model type better suited
to your use case.

**2. Adjusting Nodes**

Some common node behaviors that you might need to adjust are:

* ``focal_length`` & ``torso_factor``: We calibrated these settings using a Logitech c170 webcam,
  with 2 individuals of heights about 1.7m. We recommend running a few experiments on your setup
  and calibrate these accordingly.
* ``tag_msg``: The message to show when individuals are too close.
* ``near_threshold``: The minimum acceptable distance between 2 individuals, in meters. For
  example, if the threshold is set at 1.5m, and 2 individuals are standing 2.0m apart, ``tag_msg``
  doesn't show as they are standing further apart than the threshold. The larger this number, the
  stricter the social distancing.

For more adjustable node behaviors not listed here, check out the :ref:`API Documentation <api_doc>`.

.. _use_case_social_distancing_using_object_detection:

**3. Using Object Detection (Optional)**

It is possible to use :doc:`object detection models </resources/01a_object_detection>` instead
of pose estimation. To do so, replace the model node accordingly, and replace the
:mod:`dabble.keypoints_to_3d_loc` node with :mod:`dabble.bbox_to_3d_loc`. The reference or "ground
truth length" in this case would be the average height of a human, multiplied by a small factor.

You might need to use this approach if running on a resource-limited device such as a Raspberry Pi.
In this situation, you'll need to use the lightweight models, and we find that lightweight object
detectors are generally better than lightweight pose estimation models in detecting humans.

The trade-off here is that the estimated distance between individuals will be less accurate. This
is because for object detectors, the bounding box will be compared with the average height of a
human, but the bounding box height decreases if the person is sitting down or bending over.

Using with Group Size Checker
=============================

As part of COVID-19 measures, the Singapore Government has set restrictions on the group sizes of
social gatherings. We've developed a `group size checker <https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker>`_
that checks if the group size limit has been violated.

The nodes for group size checker can be stacked with social distancing, to perform both at the same
time. Check out the :doc:`Group Size Checking use case </use_cases/group_size_checking>` to find
out which nodes are used.
