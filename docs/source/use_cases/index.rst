:orphan:

.. include:: /include/substitution.rst

.. raw:: html

   <style type="text/css">
     div.toctree-wrapper {
       display: none;
     }

     table.docutils col {
       width: 50% !important;
     }
   </style>

*********
Use Cases
*********

Computer Vision (CV) problems come in various forms, and the gallery below shows common CV use
cases which can be tackled by PeekingDuck right out of the box. Areas include privacy protection,
smart monitoring, and COVID-19 prevention and control. Users are encouraged to mix and match
different PeekingDuck nodes and create your own :ref:`custom nodes <tutorial_custom_nodes>` for
your specific use case - the only limit is your imagination!

..
    Use case table substitutions

.. |social_distancing_doc| replace:: :doc:`Social Distancing </use_cases/social_distancing>`

.. |social_distancing_gif| image:: /assets/use_cases/social_distancing.gif
   :class: no-scaled-link
   :target: social_distancing.html
   :width: 100 %

.. |zone_counting_doc| replace:: :doc:`Zone Counting </use_cases/zone_counting>`

.. |zone_counting_gif| image:: /assets/use_cases/zone_counting.gif
   :class: no-scaled-link
   :target: zone_counting.html
   :width: 100 %

.. |group_size_checking_doc| replace:: :doc:`Group Size Checking </use_cases/group_size_checking>`

.. |group_size_checking_gif| image:: /assets/use_cases/group_size_checking.gif
   :class: no-scaled-link
   :target: group_size_checking.html
   :width: 100 %

.. |object_counting_present_doc| replace:: :doc:`Object Counting (Present) </use_cases/object_counting_present>`

.. |object_counting_present_gif| image:: /assets/use_cases/object_counting_present.gif
   :class: no-scaled-link
   :target: object_counting_present.html
   :width: 100 %

.. |privacy_protection_faces_doc| replace:: :doc:`Privacy Protection (Faces) </use_cases/privacy_protection_faces>`

.. |privacy_protection_faces_gif| image:: /assets/use_cases/privacy_protection_faces.gif
   :class: no-scaled-link
   :target: privacy_protection_faces.html
   :width: 100 %

.. |privacy_protection_lp_doc| replace:: :doc:`Privacy Protection (License Plates) </use_cases/privacy_protection_license_plates>`

.. |privacy_protection_lp_gif| image:: /assets/use_cases/privacy_protection_license_plates.gif
   :class: no-scaled-link
   :target: privacy_protection_license_plates.html
   :width: 100 %

.. |face_mask_detection_doc| replace:: :doc:`Face Mask Detection </use_cases/face_mask_detection>`

.. |face_mask_detection_gif| image:: /assets/use_cases/face_mask_detection.gif
   :class: no-scaled-link
   :target: face_mask_detection.html
   :width: 100 %

.. |crowd_counting_doc| replace:: :doc:`Crowd Counting </use_cases/crowd_counting>`

.. |crowd_counting_gif| image:: /assets/use_cases/crowd_counting.gif
   :class: no-scaled-link
   :target: crowd_counting.html
   :width: 100 %

.. |object_counting_over_time_doc| replace:: :doc:`Object Counting (Over Time) </use_cases/object_counting_over_time>`

.. |object_counting_over_time_gif| image:: /assets/use_cases/object_counting_over_time.gif
   :class: no-scaled-link
   :target: object_counting_over_time.html
   :width: 100 %

.. |people_counting_over_time_doc| replace:: :doc:`People Counting (Over Time) </use_cases/people_counting_over_time>`

.. |people_counting_over_time_gif| image:: /assets/use_cases/people_counting_over_time.gif
   :class: no-scaled-link
   :target: people_counting_over_time.html
   :width: 100 %

.. _privacy_protection_use_cases:

Privacy Protection
==================

.. toctree::
   :maxdepth: 1

   /use_cases/privacy_protection_faces
   /use_cases/privacy_protection_license_plates

+--------------------------------------------------------+-----------------------------+
| |privacy_protection_faces_doc| |tab| |tab| |tab| |tab| | |privacy_protection_lp_doc| |
+--------------------------------------------------------+-----------------------------+
| |privacy_protection_faces_gif|                         | |privacy_protection_lp_gif| |
+--------------------------------------------------------+-----------------------------+

.. _smart_monitoring_use_cases:

Smart Monitoring
================

.. toctree::
   :maxdepth: 1

   /use_cases/crowd_counting
   /use_cases/object_counting_present
   /use_cases/object_counting_over_time
   /use_cases/people_counting_over_time
   /use_cases/zone_counting

+---------------------------------+---------------------------------+
| |zone_counting_doc|             | |crowd_counting_doc|            |
+---------------------------------+---------------------------------+
| |zone_counting_gif|             | |crowd_counting_gif|            |
+---------------------------------+---------------------------------+
| |object_counting_over_time_doc| | |people_counting_over_time_doc| |
+---------------------------------+---------------------------------+
| |object_counting_over_time_gif| | |people_counting_over_time_gif| |
+---------------------------------+---------------------------------+
| |object_counting_present_doc|   |                                 |
+---------------------------------+                                 +
| |object_counting_present_gif|   |                                 |
+---------------------------------+---------------------------------+

.. _covid_19_use_cases:

COVID-19 Prevention and Control
===============================

.. toctree::
   :maxdepth: 1

   /use_cases/face_mask_detection
   /use_cases/group_size_checking
   /use_cases/social_distancing

+--------------------------------+-----------------------------+
| |social_distancing_doc|        | |group_size_checking_doc|   |
+--------------------------------+-----------------------------+
| |social_distancing_gif|        | |group_size_checking_gif|   |
+--------------------------------+-----------------------------+
| |face_mask_detection_doc|      |                             |
+--------------------------------+                             +
| |face_mask_detection_gif|      |                             |
+--------------------------------+-----------------------------+