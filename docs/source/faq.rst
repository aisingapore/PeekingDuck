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

Specifying ":term:`all <(input) all>`" as the input allows the node to receive all prior outputs as the input.
This is used by nodes such as :mod:`draw.legend` and :mod:`output.csv_writer`.

How do I debug custom nodes?
----------------------------

You can add code in custom nodes to print the contents of their inputs.
For more info, please see the tutorial on :ref:`debugging <tutorial_debugging>`.


