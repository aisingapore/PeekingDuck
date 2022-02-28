***********************
"Hello Computer Vision"
***********************

.. |br| raw:: html

   <br />

.. role:: red

.. role:: blue

.. role:: green

.. |Blank| unicode:: U+2800 .. Invisible character

.. |Enter| unicode:: U+23CE .. Unicode Enter Key Symbol

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
<verify_installation>` earlier, you are running a object detection pipeline in the file
``verification_pipeline.yml`` as shown below:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.recorded:
       input_dir: "data/verification/wave.mp4"
   - model.yolo
   - draw.bbox
   - output.screen

The above **object detection pipeline** comprises four nodes that do the following:

    2. ``input.recorded``: reads the file ``wave.mp4``, |br|
    4. ``model.yolo``: runs the ``Yolo`` object detection model on it, |br|
    5. ``draw.bbox``: draws the bounding box to show the detected person, |br|
    6. ``output.screen``: outputs everything onto the screen for display.

The 18-second video will auto-close when it is completed. |br|
To exit earlier, click to select the video window and press ``q``.

You have successfully run an object detection pipeline.

.. _tutorial_pose_estimation:

Pose Estimation
===============

To perform pose estimation with PeekingDuck, initialize the PeekingDuck project using:

.. admonition:: Terminal Session

    | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck init` \

Then, modify the ``pipeline.yml`` as follows:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.recorded:
       input_dir: "data/verification/wave.mp4"
   - model.posenet      # use pose estimation model
   - draw.poses         # draw skeletal poses
   - output.screen

The important changes are to the second node ``model`` (Line 4) and the third node ``draw`` (Line 5).

Now, run the pipeline using

.. admonition:: Terminal Session

    | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck run` \

You should see the same video with skeletal poses drawn on it and which track the hand movement.

The above **pose estimation pipeline** comprises four nodes that do the following:

    1. ``input.recorded``: reads the file ``wave.mp4``, |br|
    2. ``model.posenet``: runs the ``Posenet`` pose estimation model on it, |br|
    3. ``draw.poses``: draws the human skeletal frame to show the detected poses, |br|
    4. ``output.screen``: outputs everything onto the screen for display.

The 18-second video will auto-close when it is completed. |br|
To exit earlier, click to select the video window and press ``q``.

That's it: you have created a pose estimation pipeline by changing only two lines!

    .. note::
        Try replacing ``wave.mp4`` with your own video file and run both models. |br|
        For best effect, your video file should contain people performing some activity.


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
        See tutorial on Nodes and Configs.



