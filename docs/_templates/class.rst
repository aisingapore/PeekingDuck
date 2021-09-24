.. |img| replace:: ``img`` (:obj:`numpy.array`): a numpy array (height, width, channels)
   representation of the image, in BGR format

.. |bboxes| replace:: ``bboxes`` (:obj:`numpy.array`): a numpy array (N,4)
   containing bounding box information of detected objects. N corresponds to the number of objects detected,
   and each bounding box is represented as (x1,y1,x2,y2),
   (x,y) coordinates of the top-left corner and bottom right corner of the bounding box respectively.

.. |bbox_labels| replace:: ``bbox_labels`` (:obj:`numpy.array`): a numpy array (N) strings,
   representing the labels of detected objects. The order corresponds to `bboxes` and `bbox_scores`.

.. |bbox_scores| replace:: ``bbox_scores`` (:obj:`numpy.array`): a numpy array (N) [0,1]
   of the confidence scores for detected objects. The order corresponds to `bboxes` and `bbox_labels`.

.. |btm_midpoint| replace:: ``btm_midpoint`` (:obj:`list`): a list of tuples (x,y)
   of a single point of reference of bounding boxes for use in zone analytics. The order of btm_mipoint
   follows the order of `bboxes`

.. |count| replace:: ``count`` (:obj:`list`): A list of integers that represent the count of
   a pre-selected object (for example, people) detected in each frame through bboxes.

.. |keypoints| replace:: ``keypoints`` (:obj:`numpy.array`): a numpy array (N, K, 2) with the last
   dimension representing the (x,y) coordinates for detected poses. N represents the number of detected poses, and
   K represents individual keypoints. For keypoints that have low confidence score (below threshold),
   it will be replaced by `-1`.

.. |keypoint_scores| replace:: ``keypoints_scores`` (:obj:`numpy.array`): a numpy array (N, K, 1) containing
   the confidence scores of detected pose. N represents the number of detected poses, and K represents individual
   keypoints.

.. |keypoints_conns| replace:: ``keypoints_conns`` (:obj:`list`): a list of N numpy arrays (2, 2)
   with each array represnting one connection in the image. The numpy array contains the (x,y) coordinates
   of 2 adjacent keypoints pairs, if both keypoints are detected.

.. |pipeline_end| replace:: ``pipeline_end`` (:obj:`bool`): A boolean that evaluates as ``True`` when
   the pipeline is completed. Suitable for operations that require the complete inference pipeline to
   be completed before running.

.. |filename| replace:: ``filename`` (:obj:`str`): the filename of video/image being read.

.. |fps| replace:: ``fps`` (:obj:`list`): a list of floats, representing the FPS per frame. The
   FPS returned can either be a moving average or an instantaneous value. This setting can be changed in
   configs/dabble/fps file.

.. |saved_video_fps| replace:: ``saved_video_fps`` (:obj:`float`): the FPS of recorded video, upon filming.

.. |obj_3D_locs| replace:: ``obj_3D_locs`` (:obj:`list`): a list of numpy arrays (x,y,z) representing
   the 3D coordinates of an object associated with a detected bounding box.

.. |obj_groups| replace:: ``obj_groups`` (:obj:`list`): a list of integers, representing the assigned
   group number of an object associated with a detected bounding box.

.. |large_groups| replace:: ``large_groups`` (:obj:`list`): a list of integers, representing the group IDs
   of groups that have exceeded the size threshold.

.. |obj_tags| replace:: ``obj_tags`` (:obj:`list`): a list of strings to be added to a bounding box,
   for display. The order of the tags follow the order of "bboxes".

.. |zones| replace:: ``zones`` (:obj:`list`): A nested list of coordinates, with each sub-list containing
   the (x,y) coordinates that represent the points that form the boundaries of a zone. The order of zones
   follows the order of ``zone_counts``.

.. |zone_count| replace:: ``zone_count`` (:obj:`list`): A list of integers that represent the count of
   a pre-selected object (for example, people) detected in each specified zone. The order of counts
   follows the order of ``zones``.

.. |none| replace:: ``none`` (:obj:`none`): No inputs required, or no additional outputs produced. Used for
   input nodes that require no prior inputs, or draw nodes that overwrite current input.


{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :exclude-members: run

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   {% for item in all_methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
