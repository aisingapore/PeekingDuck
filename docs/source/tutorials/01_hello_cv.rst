***********************
"Hello Computer Vision"
***********************

.. |br| raw:: html

   <br />

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


Object Detection
================

When you did a ``peekingduck run`` to :ref:`verify your installation
<verify_installation>` earlier, you are running the default pipeline in the file
``run_config.yml`` as shown below:

.. code-block:: yaml

   nodes:
   - input.recorded:
       input_dir: "images/testing/wave.mp4"
   - model.yolo
   - draw.bbox
   - output.screen

The above **object detection pipeline** comprises four nodes that do the following:

    #. ``input.recorded``: reads the file ``wave.mp4``,

    #. ``model.yolo``: runs the ``Yolo`` object detection model on it,

    #. ``draw.bbox``: draws the bounding box to show the detected person,

    #. ``output.screen``: outputs everything onto the screen for display.

The 18-second video will auto-close when it is completed. |br|
To exit earlier, click to select the video window and press ``q``.

You have successfully run an object detection pipeline.



Pose Estimation
===============

You can get PeekingDuck to perform pose estimation by changing the second
``model`` and the third ``draw`` nodes in  ``run_config.yml`` as follows:

.. code-block:: yaml

   nodes:
   - input.recorded:
       input_dir: "images/testing/wave.mp4"
   - model.posenet      # use pose estimation model
   - draw.poses         # draw skeletal poses
   - output.screen

Now do a ``peekingduck run`` again and you will see the same video with skeletal
poses drawn on it and which track the hand movement.

The above **pose estimation pipeline** comprises four nodes that do the following:

    #. ``input.recorded``: reads the file ``wave.mp4``,

    #. ``model.posenet``: runs the ``Posenet`` pose estimation model on it,

    #. ``draw.poses``: draws the human skeletal frame to show the detected poses,

    #. ``output.screen``: outputs everything onto the screen for display.

The 18-second video will auto-close when it is completed. |br|
To exit earlier, click to select the video window and press ``q``.

That's it: you have created a pose estimation pipeline by changing only two lines!

    .. note::
        Try replacing ``wave.mp4`` with your own video file and run both models. |br|
        For best effect, your video file should contain people performing some activity.



Using a WebCam
==============

If your computer has a webcam attached, you can use it by changing the first
``input`` node as follows:

.. code-block:: yaml

   nodes:
   - input.live         # use webcam for live video
   - model.posenet      # use pose estimation model
   - draw.poses         # draw skeletal poses
   - output.screen

Now do a ``peekingduck run`` and you will see yourself onscreen. Move your hands
around and see PeekingDuck tracking your poses.

To exit, click to select the video window and press ``q``.



