***********************
"Hello Computer Vision"
***********************

.. include:: /include/substitution.rst

Computer Vision (or CV) is a field in AI that develops techniques to help
computers to "see" and "understand" the contents of digital images like
photographs and videos, and to derive meaningful information.
Common CV applications include object detection to detect what objects are
present in the image and pose estimation to detect the position of human limbs
relative to the body.

PeekingDuck allows you to build a CV pipeline to analyze and process images
and/or videos. This pipeline is made up of nodes: each node can perform certain
CV-related tasks.

This section presents two basic "hello world" examples to demonstrate how to use
PeekingDuck for object detection and pose estimation.


.. _tutorial_object_detection:

Object Detection
================

When you ran ``peekingduck --verify_install`` to :ref:`verify your installation
<verify_installation>` earlier, you were running the default pipeline in the file
``verification_pipeline.yml`` as shown below:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.recorded:
       input_dir: "data/verification/wave.mp4"
   - model.yolo
   - draw.bbox
   - output.screen

The above pipeline forms an **object detection pipeline** and comprises four nodes that do the
following:

   #. ``input.recorded``: reads the file ``wave.mp4``,
   #. ``model.yolo``: runs the Yolo object detection model on it,
   #. ``draw.bbox``: draws the bounding box to show the detected person,
   #. ``output.screen``: outputs everything onto the screen for display.

| The 18-second video will auto-close when it is completed.
| To exit earlier, click to select the video window and press ``q``.

You have successfully run an object detection pipeline.


.. _tutorial_pose_estimation:

Pose Estimation
===============

To perform pose estimation with PeekingDuck, initialize the PeekingDuck project using:

.. admonition:: Terminal Session

    | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck init` \

Then, modify the ``pipeline_config.yml`` as follows:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.recorded:
       input_dir: "data/verification/wave.mp4"
   - model.posenet      # use pose estimation model
   - draw.poses         # draw skeletal poses
   - output.screen

The important changes are to the second node ``model`` and the third node ``draw``
(Lines 4-5).

Now, run the pipeline using

.. admonition:: Terminal Session

    | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck run` \

You should see the same video with skeletal poses drawn on it and which track the hand movement.

The above **pose estimation pipeline** comprises four nodes that do the following:

   #. ``input.recorded``: reads the file ``wave.mp4``,
   #. ``model.posenet``: runs the ``Posenet`` pose estimation model on it,
   #. ``draw.poses``: draws the human skeletal frame to show the detected poses,
   #. ``output.screen``: outputs everything onto the screen for display.

| The 18-second video will auto-close when it is completed.
| To exit earlier, click to select the video window and press ``q``.

That's it: you have created a pose estimation pipeline by changing only two lines!

.. note::

   | Try replacing ``wave.mp4`` with your own video file and run both models.
   | For best effect, your video file should contain people performing some activity.


.. _tutorial_webcam:

Using a WebCam
==============

If your computer has a webcam attached, you can use it by changing the first
``input`` node (line 2) as follows:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.live         # use webcam for live video
   - model.posenet      # use pose estimation model
   - draw.poses         # draw skeletal poses
   - output.screen

Now do a ``peekingduck run`` and you will see yourself onscreen. Move your hands
around and see PeekingDuck tracking your poses.

To exit, click to select the video window and press ``q``.

.. note::

    PeekingDuck assumes the webcam is defaulted to input source 0.
    If your system is configured differently, you would have to specify the 
    input source by changing the ``input.live`` configuration.
    See tutorial on :ref:`Nodes and Configs <tutorial_nodes_config>`.


.. _tutorial_nodes_config:

Pipelines, Nodes and Configs
============================

PeekingDuck comes with a rich collection of nodes that you can use to create
your own CV pipelines. Each node can be customized by changing its
configurations or settings.

To get a quick overview of PeekingDuck's nodes, run the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`peekingduck nodes` \


.. url: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/tutorials/ss_pkd_nodes.png
.. image:: /assets/tutorials/ss_pkd_nodes.png
   :alt: PeekingDuck screenshot : nodes output

You will see a comprehensive list of all PeekingDuck's nodes with links to their
``readthedocs`` pages for more information.


PeekingDuck supports 6 types of nodes:

+-----------+-----------------------------------------------------------------+
| Node Type | Node Description                                                |
+===========+=================================================================+
| Input     | Reads a video file from disk or captures images from the webcam |
+-----------+-----------------------------------------------------------------+
| Model     | CV model does the "heaving lifting" here, like object detection |
+-----------+-----------------------------------------------------------------+
| Dabble    | Does the "smaller" computations, like counting number of bboxes |
+-----------+-----------------------------------------------------------------+
| Draw      | Draws things/text onto an image, like bboxes or FPS             |
+-----------+-----------------------------------------------------------------+
| Output    | Shows an image on screen or saves to a video file on disk       |
+-----------+-----------------------------------------------------------------+
| Augment   | Applies effects onto an image                                   |
+-----------+-----------------------------------------------------------------+

A PeekingDuck pipeline is created by stringing together a series of nodes that 
perform a logical sequence of operations.
Each node has its own set of configurable settings that can be modified to
change its behavior.


.. _tutorial_coordinate_systems:

Bounding Box vs Image Coordinates
=================================

PeekingDuck has two coordinate systems, with top-left corner as origin :math:`(0, 0)`:

   .. figure:: /assets/tutorials/bbox_image_coords.png
      :alt: Image vs Bounding Box Coordinates

      PeekingDuck's Image vs Bounding Box Coordinates

* Absolute image coordinates
   For an image of width |W| and height |H|, the absolute image coordinates are 
   integers from :math:`(0, 0)` to :math:`(W-1, H-1)`. |br|
   E.g., for a 720 x 480 image, the absolute coordinates range from 
   :math:`(0, 0)` to :math:`(719, 479)`.

* Relative bounding box coordinates
   For an image of width |W| and height |H|, the relative image coordinates are 
   real numbers from :math:`(0.0, 0.0)` to :math:`(1.0, 1.0)`. |br|
   E.g., for a 720 x 480 image, the relative coordinates range from 
   :math:`(0.0, 0.0)` to :math:`(1.0, 1.0)`.

This means that in order to draw a bounding box onto an image, the bounding box 
relative coordinates would have to be converted to the image absolute coordinates.

Using the above figure as an illustration, the bounding box coordinates are
given as :math:`(0.18, 0.10)` left-top and :math:`(0.52, 0.88)` right-bottom.
To convert them to image coordinates, multiply the x-coordinates by the image 
width and the y-coordinates by the image height, and round the results into 
integers.

.. math::

   \begin{array}{ll}
      0.18 \rightarrow 0.18 * 720 = 129.6 = 130 & (int)\\
      0.10 \rightarrow 0.10 * 720 = 72.0 = 72 & (int)\\
      &\\
      0.52 \rightarrow 0.52 * 720 = 374.4 = 374 & (int)\\
      0.88 \rightarrow 0.88 * 720 = 633.6 = 634 & (int)
   \end{array}

Thus, the image coordinates are :math:`(130, 72)` left-top and :math:`(374, 634)` right-bottom.

.. note::
   
   The ``model`` nodes return results in relative coordinates.
