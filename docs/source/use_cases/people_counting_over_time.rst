***************************
People Counting (Over Time)
***************************

Overview
========

People counting over time involves detecting and tracking different persons, and incrementing the
count when a new person appears. This use case can reduce dependency on manual counting, and be
applied to areas such as retail analytics, queue management, or occupancy monitoring. 

.. image:: /assets/use_cases/people_counting_over_time.gif
   :class: no-scaled-link
   :width: 50 %

Our solution automatically detects, tracks, and counts people over time. This is explained in the
`How it Works`_ section.

Demo
====

.. |pipeline_config| replace:: people_counting_over_time.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/docs-v1.2/use_cases/people_counting_over_time.yml

To try our solution on your own computer, :doc:`install </getting_started/02_basic_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |pipeline_config|\ >

How it Works
============

People counting over time comprises three main components:

#. Human detection,
#. Appearance embedding tracking, and
#. Incrementing the count.

**1. Human Detection**

We use an open source detection model trained on pedestrian detection and person search datasets
known as `JDE <https://arxiv.org/abs/1909.12605>`_ to identify human persons. This allows the
application to identify the locations of human persons in a video feed. Each of these locations is
represented as a pair of `x, y` coordinates in the form :math:`[x_1, y_1, x_2, y_2]`, where
:math:`(x_1, y_1)` is the top left corner of the bounding box, and :math:`(x_2, y_2)` is the bottom
right. These are used to form the bounding box of each human person detected. For more information
on how adjust the JDE node, check out the :doc:`JDE configurable parameters </nodes/model.jde>`.

**2. Appearance Embedding Tracking**

To learn appearance embeddings for tracking, a metric learning algorithm with triplet loss
together is used. Observations are assigned to tracklets using the Hungarian algorithm. The Kalman
filter is used to smooth the trajectories and predict the locations of previous tracklets in the
current frame. The model outputs an ID for each detection based on the appearance embedding learned.

**3. Incrementing the Count**

Monotonically increasing integer IDs beginning from "0" are assigned to new unique objects. For
example, the first tracked object is assigned an ID of "0", the second tracked object is assigned
an ID of "1", and so on. Thus the total number of unique objects that have appeared in the entire
duration is simply the cumulative maximum.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.recorded:
       input_dir: <path/to/video with people>
   - model.jde
   - dabble.statistics:
       maximum: obj_attrs["ids"]
   - draw.bbox
   - draw.tag:
       show: ["ids"]
   - draw.legend:
       show: ["cum_max"]
   - output.screen


**1. JDE Node**

This node employs a single network to **simultaneously** output detection results and the
corresponding appearance embeddings of the detected boxes. Therefore JDE stands for Joint Detection
and Embedding. Please take a look at the :doc:`benchmarks </resources/01c_object_tracking>` of
object tracking models that are included in PeekingDuck if you would like to use a different model
variation or an alternative model better suited to your use case.

**2. Statistics Node**

The :mod:`dabble.statistics` node retrieves the maximum detected ID for each frame. If the ID
exceeds the previous maximum, the :term:`cum_max` (cumulative maximum) is updated. As monotonically
increasing integer IDs beginning from "0" are assigned to new unique objects, the maximum ID is
equal to the total number of unique objects over time. 

**3. Adjusting Nodes**

With regard to the :mod:`model.jde` node, some common behaviors that you might want to adjust are:

* ``iou_threshold``: Specifies the threshold value for Intersection over Union of detections
  (default = 0.5). 
* ``score_threshold``: Specifies the threshold values for the detection confidence (default = 0.5).
  You may want to lower this value to increase the number of detections.
* ``nms_threshold``: Specifies the threshold value for non-maximal suppression (default = 0.4).
  You may want to lower this value to increase the number of detections.
* ``min_box_area``: Minimum value for area of detected bounding box. Calculated by :math:`width \times height`.
* ``track_buffer``: Specifies the threshold to remove track if track is lost for more
  frames than this value.

Counting People Within Zones
============================

It is possible to extend this use case with the :doc:`Zone Counting </use_cases/zone_counting>`
use case. For example, if a CCTV footage shows the entrance of a mall as well as a road, and we are
only interested to apply people counting to the mall entrance, we could split the video into 2
different zones and only count the people within the chosen zone. An example of how this can be done
is given in the :ref:`Tracking People within a Zone <tutorial_tracking_within_zone>` tutorial.