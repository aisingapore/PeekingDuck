.. include:: /include/data_type.rst
.. include:: /include/substitution.rst

********
Glossary
********

The following are built-in data types recognized by PeekingDuck nodes. Users can define custom data
types when working with custom nodes.

.. glossary::

   (input) |all|
      |all_input_def|
   
   |bboxes|
      |bboxes_def|

   |bbox_labels|
      |bbox_labels_def|

   |bbox_scores|
      |bbox_scores_def|
   
   |btm_midpoint|
      |btm_midpoint_def|
   
   |count|
      |count_def|
   
   |cum_avg|
      |cum_avg_def|
   
   |cum_max|
      |cum_max_def|
   
   |cum_min|
      |cum_min_def|
   
   |density_map|
      |density_map_def|
   
   |filename|
      |filename_def|
   
   |fps|
      |fps_def|
   
   |img|
      |img_def|
   
   |keypoints|
      |keypoints_def|
   
   |keypoint_conns|
      |keypoint_conns_def|
   
   |keypoint_scores|
      |keypoint_scores_def|
   
   |large_groups|
      |large_groups_def|
   
   (input) |none|
      |none_input_def|
   
   (output) |none|
      |none_output_def|
   
   |obj_3D_locs|
      |obj_3D_locs_def|
   
   |obj_attrs|
      |obj_attrs_def|
   
   |pipeline_end|
      |pipeline_end_def|
   
   |saved_video_fps|
      |saved_video_fps_def|
   
   |zones|
      |zones_def|
   
   |zone_count|
      |zone_count_def|

.. deprecated:: 1.2.0
    ``obj_tags`` (:obj:`List[str]`) is deprecated and now subsumed under
    :term:`obj_attrs`. :mod:`dabble.check_nearby_objs` now accesses this
    attribute by using the ``flags`` key of :term:`obj_attrs`. :mod:`draw.tag`
    has been refactored for more drawing flexibility by accepting
    :term:`obj_attrs` as input.

.. deprecated:: 1.2.0
    ``obj_groups`` (:obj:`List[int]`) is deprecated and now subsumed under 
    :term:`obj_attrs`. Affected nodes (:mod:`dabble.group_nearby_objs`, 
    :mod:`dabble.check_large_groups`, and :mod:`draw.group_bbox_and_tag`) now
    access this attribute by using the ``groups`` key of :term:`obj_attrs`.
