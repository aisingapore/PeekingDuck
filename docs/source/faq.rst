FAQ and Troubleshooting
=======================

How can I post-process and visualize model outputs?
---------------------------------------------------

The common output of all ``model`` nodes is :term:`bboxes`. :term:`bboxes` can
be used for subsequent actions like counting (:mod:`dabble.bbox_count`), drawing
(:mod:`draw.bbox`), tagging (:mod:`draw.tag`), etc. You can also create custom
nodes which takes :term:`bboxes` as an input to visualize your results.

How can I dynamically use all prior outputs as the input at run time?
---------------------------------------------------------------------

Specifying ":term:`all <(input) all>`" as the input allows the node to receive all prior
outputs as the input.
This is used by nodes such as :mod:`draw.legend` and :mod:`output.csv_writer`.

How do I debug custom nodes?
----------------------------

You can add code in custom nodes to print the contents of their inputs.
For more info, please see the tutorial on :ref:`debugging <tutorial_debugging>`.

Why does :mod:`input.visual` progress update not reach 100%?
------------------------------------------------------------

:mod:`input.visual` provides progress update information for inputs if it is able to get
a total frame count.
This number is obtained through ``opencv``'s API which attempts to estimate the total 
frame count using the media's metadata duration and FPS.
It is not an accurate number but an estimate, as it is affected by potential errors such
as frame corruption, inaccurate FPS, rounding errors, and others.
