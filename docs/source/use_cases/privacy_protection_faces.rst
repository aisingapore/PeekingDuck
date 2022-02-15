**************************
Privacy Protection (Faces)
**************************

Overview
========

As organizations collect more data, there is a need to better protect the identities of individuals
in public and private places. AI Singapore has developed a solution that performs face
anonymization. This can be used to comply with the General Data Protection Regulation (GDPR) or
other data privacy laws.

.. image:: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/privacy_protection_faces.gif
   :class: no-scaled-link
   :width: 100 %

Our solution automatically detects and mosaics (or blurs) human faces. This is explained in the
`How it Works`_ section.

Demo
====

.. |run_config| replace:: privacy_protection_faces.yml
.. _run_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_protection_faces.yml

To try our solution on your own computer, :doc:`install </getting_started/01_basic_install>` and run
PeekingDuck with the configuration file |run_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |run_config|\ >

How it Works
============

There are two main components to face anonymization:

#. Face detection using AI and
#. Face de-identification. 

**1. Face Detection**

We use an open source face detection model known as `MTCNN <https://arxiv.org/abs/1604.02878>`_ to
identify human faces. This allows the application to identify the locations of human faces in a
video feed. Each of these locations is represented as a pair of `x, y` coordinates in the form
:math:`[x_1, y_1, x_2, y_2]`, where :math:`(x_1, y_1)` is the top left corner of the bounding box,
and :math:`(x_2, y_2)` is the bottom right. These are used to form the bounding box of each human
face detected. For more information on how adjust the MTCNN node, check out the
:doc:`MTCNN configurable parameters </nodes/model.mtcnn>`.

**2. Face De-Identification**

To perform face de-identification, we pixelate or gaussian blur the areas bounded by the bounding
boxes.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |run_config|_):

.. code-block:: yaml

   nodes:
   - input.live
   - model.mtcnn
   - dabble.fps
   - draw.mosaic_bbox
   - draw.legend
   - output.screen


**1. Face Detection Node**

As mentioned, we use the MTCNN model for face detection. It is able to detect human faces with face
masks to a certain extent. Please take a look at the :doc:`benchmarks </resources/01a_object_detection>`
of object detection models that are included in PeekingDuck if you would like to use a different
model variation or an alternative model better suited to your use case.

**2. Face De-Identification Nodes**

You can mosaic or blur the faces detected using the :mod:`draw.mosaic_bbox` or
:mod:`draw.blur_bbox` in the run config declaration.

**3. Adjusting Nodes**

With regard to the MTCNN model, some common node behaviors that you might want to adjust are:

* ``mtcnn_min_size``: Specifies in pixels the minimum height and width of a face to be detected.
  (default = 40) You may want to decrease the minimum size to increase the number of detections.
* ``mtcnn_thresholds``: This specifies the threshold values for the Proposal Network (P-Net),
  Refine Network (R-Net), and Output Network (O-Net) in the MTCNN model. (default = [0.6, 0.7, 0.7])
  Calibration is performed at each stage in which bounding boxes with confidence scores less than
  the specified threshold are discarded. 
* ``mtcnn_score``: Specifies the threshold value in the final output. (default = 0.7) Bounding
  boxes with confidence scores less than the specified threshold in the final output are discarded.
  You may want to lower ``mtcnn_thresholds`` and ``mtcnn_score`` to increase the number of
  detections.

In addition, some common node behaviors that you might want to adjust for the
:mod:`dabble.mosaic_bbox` and :mod:`dabble.blur_bbox` nodes are:

* ``mosaic_level``: Defines the resolution of a mosaic filter (:math:`width \times height`); the
  value corresponds to the number of rows and columns used to create a mosaic. (default = 7) For
  example, the default value creates a :math:`7 \times 7` mosaic filter. Increasing the number
  increases the intensity of pixelation over an area.
* ``blur_level``:  Defines the standard deviation of the Gaussian kernel used in the Gaussian
  filter. (default = 50) The higher the blur level, the more intense is the blurring.