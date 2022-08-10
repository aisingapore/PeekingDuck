**********************************************
Privacy Protection (People & Computer Screens)
**********************************************

Overview
========

Videos and pictures often contain people and other sensitive visual information (e.g., the display
on computer screens), even though this information might not be needed at all for visual
processing. Our solution performs full body anonymization and computer screen blurring to protect
the identities of individuals and the sensitive information on computer screens. It can be used to comply with the General Data Protection Regulation (GDPR) or other data
privacy laws.

.. image:: /assets/use_cases/privacy_protection_people_screens.gif
   :class: no-scaled-link
   :width: 50 %

In this example use case, we want to count the number of people in the office, but also want to avoid
compromising the privacy of the office inhabitants or information displayed on computer screens.

Our solution automatically detects people, laptop and computer screens, and then blurs them. This is
explained in the `How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: privacy_protection_people_screens.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/main/use_cases/privacy_protection_people_screens.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

There are 2 main components to our solution:

#. Person and computer screen segmentation, and
#. Person and computer screen blurring.

**1. Person and Computer Screen Segmentation**

We use an open source instance segmentation model known as `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_
to obtain the masks of persons, computer screens and laptops. The masks are akin to the input frames
or images, except that it only has a single channel and each pixel on the mask is a binary of either
1 or 0, which indicates whether a specific class of thing is present (1) or absent (0) in a
particular location of the image. For more information on how to adjust the ``mask_rcnn`` node, check
out its :doc:`configurable parameters </nodes/model.mask_rcnn>`.

**2. Person and Computer Screen Blurring**

To blur the people and computer screens, we pixelate or gaussian blur the image pixels where the
pixel values of the relevant masks are equal to 1 (indicating presence of object).

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

    nodes:
    - input.visual:
        source: <path/to/video>
    - model.mask_rcnn:
        detect: ["tv", "laptop"]
    - draw.instance_mask:
        effect: {blur: 50}
    - model.mask_rcnn:
        detect: ["person"]
    - dabble.bbox_count
    - draw.instance_mask:
        effect: {blur: 50}
    - draw.bbox:
        show_labels: True
    - draw.legend:
        show: ["count"]
    - output.screen


*This config includes the use of two model.mask_rcnn and draw.instance_mask nodes so that the detected instances of 
"tv" and "laptop" classes can be separated from the "person" class, such that drawing and counting of bboxes are only 
performed on the "person" class*

**1. Instance Segmentation Node**

In this example use case, we used the Mask R-CNN model for instance segmentation. It can detect
persons as well as computer monitors. Please take a look at the :doc:`benchmarks </resources/01e_instance_segmentation>`
of instance segmentation models that are included in PeekingDuck if you would like to use a different
model or model type better suited to your use case.

**2. People and Screens De-Identification Node**

The detected people and screens are blurred using the :mod:`draw.instance_mask` node.

**3. Object Counting Node**

:mod:`dabble.bbox_count` counts the total number of detected bounding boxes. This node has no
configurable parameters.

**4. Display Bounding Box Node**

Then we draw bounding boxes around the detected persons using the :mod:`draw.bbox` node.

**5. Person Count Display Node**

The total number of detected persons are shown using the :mod:`draw.legend` node.

**6. Adjusting Nodes**

With regard to the Mask R-CNN model, some common node behaviors that you might want to adjust are:

* ``model_type``: Defines the type of backbones to be used.
* ``score_threshold``: Bounding boxes with classification score below the threshold will be discarded.
* ``mask_threshold``: The confidence threshold for binarizing the masks' pixel values, whether an
  object is detected at a particular pixel.

In addition, some common node behaviors that you might want to adjust for the
:mod:`draw.instance_mask` node are:

* ``blur``:  Blurs the area using this value as the “blur_kernel_size” parameter. Larger values
  gives more intense blurring.
* ``mosaic``: Mosaics the area using this value as the resolution of a mosaic filter (:math:`width \times height`).
  The value corresponds to the number of rows and columns used to create a mosaic. For example,
  the setting (``mosaic: 25``) creates a :math:`25 \times 25` mosaic filter. Increasing the number
  increases the intensity of pixelation over an area.
