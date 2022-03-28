**************************
Privacy Protection (Faces)
**************************

Overview
========

As organizations collect more data, there is a need to better protect the identities of individuals
in public and private places. Our solution performs face anonymization, and can be used to comply
with the General Data Protection Regulation (GDPR) or other data privacy laws.

.. image:: /assets/use_cases/privacy_protection_faces.gif
   :class: no-scaled-link
   :width: 50 %

Our solution automatically detects and mosaics (or blurs) human faces. This is explained in the
`How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: privacy_protection_faces.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_protection_faces.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

There are two main components to face anonymization:

#. Face detection, and
#. Face de-identification. 

**1. Face Detection**

We use an open source face detection model known as `MTCNN <https://arxiv.org/abs/1604.02878>`_ to
identify human faces. This allows the application to identify the locations of human faces in a
video feed. Each of these locations is represented as a pair of `x, y` coordinates in the form
:math:`[x_1, y_1, x_2, y_2]`, where :math:`(x_1, y_1)` is the top left corner of the bounding box,
and :math:`(x_2, y_2)` is the bottom right. These are used to form the bounding box of each human
face detected. For more information on how to adjust the MTCNN node, check out the
:doc:`MTCNN configurable parameters </nodes/model.mtcnn>`.

**2. Face De-Identification**

To perform face de-identification, we pixelate or gaussian blur the areas bounded by the bounding
boxes.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.visual:
       source: 0
   - model.mtcnn
   - draw.mosaic_bbox
   - output.screen


**1. Face Detection Node**

As mentioned, we use the MTCNN model for face detection. It is able to detect human faces with face
masks. Please take a look at the :doc:`benchmarks </resources/01a_object_detection>`
of object detection models that are included in PeekingDuck if you would like to use a different
model or model type better suited to your use case.

**2. Face De-Identification Nodes**

You can mosaic or blur the faces detected using the :mod:`draw.mosaic_bbox` or
:mod:`draw.blur_bbox` in the run config declaration.

.. figure:: /assets/use_cases/privacy_protection_faces_comparison.jpg
   :alt: De-identification effect comparison
   :class: no-scaled-link
   :width: 50 %

   De-identification with mosaic (left) and blur (right).

**3. Adjusting Nodes**

With regard to the MTCNN model, some common node behaviors that you might want to adjust are:

* ``min_size``: Specifies in pixels the minimum height and width of a face to be detected.
  (default = 40) You may want to decrease the minimum size to increase the number of detections.
* ``network_thresholds``: Specifies the threshold values for the Proposal Network (P-Net),
  Refine Network (R-Net), and Output Network (O-Net) in the MTCNN model. (default = [0.6, 0.7, 0.7])
  Calibration is performed at each stage in which bounding boxes with confidence scores less than
  the specified threshold are discarded. 
* ``score_threshold``: Specifies the threshold value in the final output. (default = 0.7) Bounding
  boxes with confidence scores less than the specified threshold in the final output are discarded.
  You may want to lower ``network_thresholds`` and ``score_threshold`` to increase the number of
  detections.

In addition, some common node behaviors that you might want to adjust for the
:mod:`dabble.mosaic_bbox` and :mod:`dabble.blur_bbox` nodes are:

* ``mosaic_level``: Defines the resolution of a mosaic filter (:math:`width \times height`); the
  value corresponds to the number of rows and columns used to create a mosaic. (default = 7) For
  example, the default value creates a :math:`7 \times 7` mosaic filter. Increasing the number
  increases the intensity of pixelization over an area.
* ``blur_level``:  Defines the standard deviation of the Gaussian kernel used in the Gaussian
  filter. (default = 50) The higher the blur level, the greater the blur intensity.
