**************
Crowd Counting
**************

.. include:: /include/substitution.rst

Overview
========

In Computer Vision (CV), crowd counting refers to the technique of counting or estimating the 
number of people in a crowd. This can be used to estimate the number of people attending an
event, monitor crowd levels and prevent human stampedes.

.. image:: /assets/use_cases/crowd_counting.gif
   :class: no-scaled-link
   :width: 50 %

Our solution utilizes CSRNet to estimate the size of a crowd. In addition, it generates a heat map
that can be used to pinpoint possible bottlenecks at a venue. This is explained in the 
`How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: crowd_counting.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/crowd_counting.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

You may like to try it on this `sample video <https://storage.googleapis.com/peekingduck/videos/large_crowd.mp4>`_.

How It Works
============

There are two main components to our solution:

#. Crowd counting, and
#. Heat map generation.

**1. Crowd Counting**

We use an open source crowd counting model known as `CSRNet <https://arxiv.org/pdf/1802.10062.pdf>`_
to predict the number of people in a sparse or dense crowd. The solution uses the
sparse crowd model by default and can be configured to use the dense crowd model if required. The
dense and sparse crowd models were trained using data from `ShanghaiTech <https://github.com/desenzhou/ShanghaiTechDataset>`_
Part A and Part B respectively.

As a rule of thumb, you might want to use the dense crowd model if the people in a given image or
video frame are packed shoulder to shoulder, e.g., stadiums. For more information on how to adjust the
CSRNet node, check out its :doc:`configurable parameters </nodes/model.csrnet>`.

**2. Heat Map Generation (Optional)**

We generate a heat map using the density map estimated by the model. Areas that are more crowded
are highlighted in red while areas that are less crowded are highlighted in blue.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.visual:
       source: <path/to/video with crowd>
   - model.csrnet:
       model_type: dense
   - draw.heat_map
   - draw.legend:
       show: ["count"]
   - output.screen

**1. Crowd Counting Node**

As mentioned, we use CSRNet to estimate the size of a crowd. As the models were trained to
recognize congested scenes, the estimates are less accurate if the number of people is low, i.e.,
below ten. In such scenarios, you should consider using the
:doc:`object detection models </resources/01a_object_detection>` included in our repo.

**2. Heat Map Generation Node (Optional)**

The heat map generation node superimposes a heat map over a given image or video frame.

**3. Adjusting Nodes**

Some common node behaviors that you might want to adjust are:

* ``model_type``: This specifies the model to be used, i.e., ``sparse`` or ``dense``. By default,
  our solution uses the sparse crowd model. As a rule of thumb, you might want to use the dense
  crowd model if the people in a given image or video frame are packed shoulder to shoulder, e.g.,
  stadiums.
* ``width``: This specifies the input width. By default, the width of an image will be resized
  to 640 for inference. The height of the image will be resized proportionally to preserve its
  aspect ratio. In general, decreasing the width of an image will improve inference speed. However,
  this might impact the accuracy of the model.
