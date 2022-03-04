.. include:: /include/data_type.rst
.. include:: /include/substitution.rst

Glossary
========

The following are built-in data types recognized by PeekingDuck nodes. Users can define custom data
types when working with custom nodes.

(input) |all_input|

|bboxes|

|bbox_labels|

|bbox_scores|

|btm_midpoint|

|count|

|cum_avg|

|cum_max|

|cum_min|

|density_map|

|filename|

|fps|

|img|

|keypoints|

|keypoint_conns|

|keypoint_scores|

|large_groups|

(input) |no_input|

(output) |no_output|

|obj_3D_locs|

|obj_attrs|

|pipeline_end|

|saved_video_fps|

|zones|

|zone_count|

.. deprecated:: 1.2.0 |br|
    ``obj_tags`` (:obj:`List[str]`) is deprecated and now subsumed under
    ``obj_attrs`` (:obj:`Dict[str, Any]`). :mod:`dabble.check_nearby_objs` now accesses
    this attribute by the ``flags`` key of ``obj_attrs``. :mod:`draw.tag` has been refactored
    for more drawing flexibility by accepting ``obj_attrs`` as input. |br|

    ``obj_groups`` (:obj:`List[int]`) is deprecated and now subsumed under 
    ``obj_attrs`` (:obj:`Dict[str, Any]`). Affected nodes (:mod:`dabble.group_nearby_objs`, 
    :mod:`dabble.check_large_groups`, :mod:`draw.group_bbox_and_tag`) now access this attribute 
    by the ``groups`` key of ``obj_attrs``.
