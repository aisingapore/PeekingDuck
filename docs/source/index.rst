************
Introduction
************

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg
   :target: https://pypi.org/project/peekingduck

.. image:: https://badge.fury.io/py/peekingduck.svg
   :target: https://pypi.org/project/peekingduck

.. image:: https://img.shields.io/pypi/dm/peekingduck
   :target: https://pypi.org/project/peekingduck
    
.. image:: https://img.shields.io/badge/license-Apache%202.0-blue.svg
   :target: https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE


What is PeekingDuck?
====================

PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV)
inference. It helps to significantly cut down development time when building CV pipelines. The name
"PeekingDuck" is a play on these words: "Peeking" in a nod to CV; and "Duck" in
`duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_ used in Python.


.. _how_peekingduck_works:

How PeekingDuck Works
=====================

**Nodes** are the building blocks of PeekingDuck. Each node is a wrapper for a Python function, and contains information on how other PeekingDuck nodes may interact with it.

PeekingDuck has 6 types of nodes:

.. image:: /assets/diagrams/node_types.drawio.svg

A **pipeline** governs the behavior of a chain of nodes. The diagram below shows the pipeline used
in the previous demo. Nodes in a pipeline are called in sequential order, and the output of one
node will be the input to another. For example, :mod:`input.live` produces ``img``, which is taken
in by :mod:`model.yolo`, and :mod:`model.yolo` produces ``bboxes``, which is taken in by
:mod:`draw.bbox`. For ease of visualization, not all the inputs and outputs of these nodes are
included in this diagram.

.. image:: /assets/diagrams/yolo_demo.drawio.svg


Explore PeekingDuck's Features
==============================

The rest of this webpage contains more details about using PeekingDuck, including information on:

* :ref:`Changing PeekingDuck nodes <tutorial_configure_nodes>` and their settings
* :ref:`API Documentation <api_doc>` for all PeekingDuck nodes, describing their behavior, inputs,
  outputs and settings
* Creating your own :ref:`custom nodes <tutorial_custom_nodes>`, and using them with
  PeekingDuck nodes
* Using PeekingDuck as an :ref:`imported Python module <tutorial_import_peekingduck>` in
  your project
* Benchmarks and class/keypoints IDs for :doc:`object detection </resources/01a_object_detection>` and
  :doc:`pose estimation </resources/01b_pose_estimation>` models.

You are also welcome to join discussions about using PeekingDuck in the following channels:

* `Github discussion board <https://github.com/aimakerspace/PeekingDuck/discussions>`_
* `AI Singapore's Community forum <https://community.aisingapore.org/groups/computer-vision/forum>`_

PeekingDuck Use Cases
=====================

AI models are cool and fun, but we're also interested in using them to solve real-world problems.
We've combined :ref:`dabble <api_doc>` nodes with :ref:`model <api_doc>` nodes to create
**use cases**, such as `social distancing <https://aisingapore.org/2020/06/hp-social-distancing>`_
and `group size checking <https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker>`_
to help combat COVID-19. For more details, click on the heading of each use case below.

..
    Use case table substitutions

.. |social_distancing_doc| replace:: :doc:`Social Distancing </use_cases/social_distancing>`

.. |social_distancing_gif| image:: /assets/use_cases/social_distancing.gif
   :class: no-scaled-link
   :width: 100 %

.. |zone_counting_doc| replace:: :doc:`Zone Counting </use_cases/zone_counting>`

.. |zone_counting_gif| image:: /assets/use_cases/zone_counting.gif
   :class: no-scaled-link
   :width: 100 %

.. |group_size_checking_doc| replace:: :doc:`Group Size Checking </use_cases/group_size_checking>`

.. |group_size_checking_gif| image:: /assets/use_cases/group_size_check_2.gif
   :class: no-scaled-link
   :width: 100 %

.. |object_counting_doc| replace:: :doc:`Object Counting </use_cases/object_counting>`

.. |object_counting_gif| image:: /assets/use_cases/object_counting.gif
   :class: no-scaled-link
   :width: 100 %

.. |privacy_protection_faces_doc| replace:: :doc:`Privacy Protection (Faces) </use_cases/privacy_protection_faces>`

.. |privacy_protection_faces_gif| image:: /assets/use_cases/privacy_protection_faces.gif
   :class: no-scaled-link
   :width: 100 %

.. |privacy_protection_lp_doc| replace:: :doc:`Privacy Protection (License Plates) </use_cases/privacy_protection_license_plates>`

.. |privacy_protection_lp_gif| image:: /assets/use_cases/privacy_protection_license_plates.gif
   :class: no-scaled-link
   :width: 100 %

.. |face_mask_detection_doc| replace:: :doc:`Face Mask Detection </use_cases/face_mask_detection>`

.. |face_mask_detection_gif| image:: /assets/use_cases/mask_detection.gif
   :class: no-scaled-link
   :width: 100 %

.. |crowd_counting_doc| replace:: :doc:`Crowd Counting </use_cases/crowd_counting>`

.. |crowd_counting_gif| image:: /assets/use_cases/crowd_counting.gif
   :class: no-scaled-link
   :width: 100 %

.. |multiple_object_tracking_doc| replace:: :doc:`Multiple Object Tracking </use_cases/multiple_object_tracking>`

.. |multiple_object_tracking_gif| image:: /assets/use_cases/vehicles_tracking.gif
   :class: no-scaled-link
   :width: 100 %

.. |human_tracking_doc| replace:: :doc:`Human Detection and Tracking </use_cases/human_tracking>`

.. |human_tracking_gif| image:: /assets/use_cases/human_tracking.gif
   :class: no-scaled-link
   :width: 100 %

+--------------------------------+-----------------------------+
| |social_distancing_doc|        | |zone_counting_doc|         |
+--------------------------------+-----------------------------+
| |social_distancing_gif|        | |zone_counting_gif|         |
+--------------------------------+-----------------------------+
| |group_size_checking_doc|      | |object_counting_doc|       |
+--------------------------------+-----------------------------+
| |group_size_checking_gif|      | |object_counting_gif|       |
+--------------------------------+-----------------------------+
| |privacy_protection_faces_doc| | |privacy_protection_lp_doc| |
+--------------------------------+-----------------------------+
| |privacy_protection_faces_gif| | |privacy_protection_lp_gif| |
+--------------------------------+-----------------------------+
| |face_mask_detection_doc|      | |crowd_counting_doc|        |
+--------------------------------+-----------------------------+
| |face_mask_detection_gif|      | |crowd_counting_gif|        |
+--------------------------------+-----------------------------+
| |multiple_object_tracking_doc| | |human_tracking_doc|        |
+--------------------------------+-----------------------------+
| |multiple_object_tracking_gif| | |human_tracking_gif|        |
+--------------------------------+-----------------------------+

We're constantly developing new nodes to increase PeekingDuck's capabilities. You've gotten a taste
of some of our commonly used nodes in the previous demos, but PeekingDuck can do a lot more. To see
what other nodes are available, check out PeekingDuck's :ref:`API Documentation <api_doc>`.

Acknowledgements
================

This project is supported by the National Research Foundation, Singapore under its AI Singapore
Programme (AISG-RP-2019-050). Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and do not reflect the views of National Research
Foundation, Singapore.


License
=======

PeekingDuck is under the open source `Apache License 2.0 <https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE>`_ (:

Even so, your organization may require legal proof of its right to use PeekingDuck, due to
circumstances such as the following:

* Your organization is using PeekingDuck in a jurisdiction that does not recognize this license
* Your legal department requires a license to be purchased
* Your organization wants to hold a tangible legal document as evidence of the legal right to use
  and distribute PeekingDuck

`Contact us <https://aisingapore.org/home/contact>`_ if any of these circumstances apply to you.
