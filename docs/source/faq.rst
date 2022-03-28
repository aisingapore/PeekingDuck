***********************
FAQ and Troubleshooting
***********************


How can I post-process and visualize model outputs?
---------------------------------------------------

The common output of all ``model`` nodes is :term:`bboxes`. :term:`bboxes` can
be used for subsequent actions like counting (:mod:`dabble.bbox_count`), drawing
(:mod:`draw.bbox`), tagging (:mod:`draw.tag`), etc. You can also create custom
nodes which take :term:`bboxes` as an input to visualize your results.


How can I dynamically use all prior outputs as the input at run time?
---------------------------------------------------------------------

Specifying ":term:`all <(input) all>`" as the input allows the node to receive all prior
outputs as the input.
This is used by nodes such as :mod:`draw.legend` and :mod:`output.csv_writer`.


How do I debug custom nodes?
----------------------------

You can add code in custom nodes to print the contents of their inputs.
For more info, please see the tutorial on :ref:`debugging <tutorial_debugging>`.


Why does :mod:`input.visual` progress stop before 100%?
-------------------------------------------------------

:mod:`input.visual` provides progress information if it is able to get a total frame
count for the input.
This number is obtained using ``opencv``'s ``CV_CAP_PROP_FRAME_COUNT`` API, which
attempts to estimate the total frame count using the input media's metadata duration and
FPS.
However, the total frame count is only an estimate.
It is not guaranteed to be accurate because it is affected by potential errors, such as
frame corruption, video decoder failure, inaccurate FPS, and rounding errors.


Why does the output screen flash briefly and disappear on my second run?
-------------------------------------------------------------------------

If you are running PeekingDuck on the Windows Subsystem for Linux (WSL), this erroneous behavior
may be caused by a WSL bug where the key buffer is not flushed. Please refer to this
`GitHub issue <https://github.com/aimakerspace/PeekingDuck/issues/630>`_ for more details.