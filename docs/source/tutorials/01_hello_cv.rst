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
PeekingDuck for pose estimation and object detection.


.. _tutorial_pose_estimation:

Pose Estimation
===============

To perform pose estimation with PeekingDuck, initialize a new PeekingDuck project using
the following commands:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`mkdir pose_estimation` \
    | \ :blue:`[~user]` \ > \ :green:`cd pose_estimation` \
    | \ :blue:`[~user/pose_estimation]` \ > \ :green:`peekingduck init` \

:greenbox:`peekingduck init` will prepare the ``pose_estimation`` folder for use with
PeekingDuck.
It creates a default pipeline file called ``pipeline_config.yml`` and a ``src`` folder
that will be covered in the later tutorials.
The ``pipeline_config.yml`` file looks like this:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.visual:
       source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
   - model.posenet
   - draw.poses
   - output.screen

The above forms a **pose estimation pipeline** and it comprises four nodes that do the
following:

   #. :mod:`input.visual`: reads the file ``wave.mp4`` from PeekingDuck's cloud storage
   #. :mod:`model.posenet`: runs the PoseNet pose estimation model on it
   #. :mod:`draw.poses`: draws a human pose skeleton over the person tracking his hand movement
   #. :mod:`output.screen`: outputs everything onto the screen for display

Now, run the pipeline using

.. admonition:: Terminal Session

    | \ :blue:`[~user/pose_estimation]` \ > \ :green:`peekingduck run` \

| You should see the following video of a person waving his hand.
| Skeletal poses are drawn on him which track the hand movement.

   .. figure:: /assets/tutorials/ss_pose_estimation.png
      :width: 389
      :alt: Pose Estimation Screenshot

      PeekingDuck's Pose Estimation Screenshot

You have successfully run a PeekingDuck pose estimation pipeline!

| The video will auto-close when it is completed.
| To exit earlier, click to select the video window and press :greenbox:`q`.


.. _tutorial_object_detection:

Object Detection
================

To perform object detection, initialize a new PeekingDuck project using the following
commands:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`mkdir object_detection` \
    | \ :blue:`[~user]` \ > \ :green:`cd object_detection` \
    | \ :blue:`[~user/object_detection]` \ > \ :green:`peekingduck init` \

Then modify ``pipeline_config.yml`` as follows:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.visual:
       source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
   - model.yolo
   - draw.bbox
   - output.screen

The key differences between this and the earlier pipeline are:

   | Line 4: :mod:`model.yolo` runs the YOLO object detection model
   | Line 5: :mod:`draw.bbox` draws the bounding box to show the detected person

Run the new **object detection pipeline** with :greenbox:`peekingduck run`.

You will see the same video with a bounding box surrounding the person.

   .. figure:: /assets/tutorials/ss_object_detection.png
      :width: 389
      :alt: Object Detection Screenshot

      PeekingDuck's Object Detection Screenshot

That's it: you have created a new object detection pipeline by changing only two lines!

   .. note::

      | Try replacing ``wave.mp4`` with your own video file and run both models.
      | For best effect, your video file should contain people performing some activities.


.. _tutorial_webcam:

Using a WebCam
==============

If your computer has a webcam attached, you can use it by changing the first
``input`` node (line 2) as follows:

.. code-block:: yaml
   :linenos:

   nodes:
   - input.visual:
       source: 0        # use webcam for live video
   - model.posenet      # use pose estimation model
   - draw.poses         # draw skeletal poses
   - output.screen

Now do a :greenbox:`peekingduck run` and you will see yourself onscreen. 
Move your hands around and see PeekingDuck tracking your poses.

To exit, click to select the video window and press :greenbox:`q`.

   .. note::

      PeekingDuck assumes the webcam is defaulted to input source 0.
      If your system is configured differently, you would have to specify the 
      input source by changing the :mod:`input.visual` configuration.
      See :ref:`changing node configuration <tutorial_more_object_detection>`.


.. _tutorial_pipeline_nodes_configs:

Pipelines, Nodes and Configs
============================

PeekingDuck comes with a rich collection of nodes that you can use to create
your own CV pipelines. Each node can be customized by changing its
configurations or settings.

To get a quick overview of PeekingDuck's nodes, run the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`peekingduck nodes` \


.. url: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/main/images/tutorials/ss_pkd_nodes.png
.. image:: /assets/tutorials/ss_pkd_nodes.png
   :alt: PeekingDuck screenshot : nodes output

You will see a comprehensive list of all PeekingDuck's nodes with links to their
``readthedocs`` pages for more information.


PeekingDuck supports 6 types of nodes:

.. image:: /assets/diagrams/node_types.drawio.svg
   :class: no-scaled-link
   :width: 60%

.. _tutorial_pipeline_data_pool:

A PeekingDuck pipeline is created by stringing together a series of nodes that 
perform a logical sequence of operations.
Each node has its own set of configurable settings that can be modified to
change its behavior.
An example pipeline is shown below:

.. image:: /assets/diagrams/yolo_demo.drawio.svg


.. _tutorial_coordinate_systems:

Bounding Box vs Image Coordinates
=================================

PeekingDuck has two :math:`(x, y)` coordinate systems, with top-left corner as origin
:math:`(0, 0)`:

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
given as :math:`(0.18, 0.10)` top-left and :math:`(0.52, 0.88)` bottom-right.
To convert them to image coordinates, multiply the x-coordinates by the image 
width and the y-coordinates by the image height, and round the results into 
integers.

.. math::

   \begin{array}{ll}
      0.18 \rightarrow 0.18 \times 720 = 129.6 = 130 & (int)\\
      0.10 \rightarrow 0.10 \times 480 = 48.0 = 48 & (int)\\
      &\\
      0.52 \rightarrow 0.52 \times 720 = 374.4 = 374 & (int)\\
      0.88 \rightarrow 0.88 \times 480 = 422.4 = 422 & (int)
   \end{array}

Thus, the image coordinates are :math:`(130, 48)` top-left and :math:`(374, 422)` bottom-right.

.. note::
   
   The :mod:`model` nodes return results in relative coordinates.
