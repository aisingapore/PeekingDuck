.. |img| replace:: ``img`` (:obj:`numpy.array`): a numpy array (height, width, channels)
   representation of the image, in BGR format

.. |bboxes| replace:: ``bboxes`` (:obj:`numpy.array`): a numpy array (N,4)
   containing bounding box information of detected objects. N corresponds to the number of objects detected,
   and each bounding box is represented as (x1,y1,x2,y2),
   (x,y) coordinates of the top-left corner and bottom right corner of the bounding box respectively.

.. |bbox_labels| replace:: ``bbox_labels`` (:obj:`numpy.array`): a numpy array (N) strings,
   representing the labels of detected objects. The order corresponds to `bboxes` and `bbox_scores`.

.. |bbox_scores| replace:: ``bbox_scores`` (:obj:`numpy.array`): a numpy array (N). [0,1]
   of the confidence scores for detected objects. The order corresponds to `bboxes` and `bbox_labels`.

.. |keypoints| replace:: ``keypoints`` (:obj:`numpy.array`): a numpy array (N, keypoints, 2) containing the
   xy coordinates for detected poses. N represents the number of detected poses, and K represents indiviudal
   keypoints. For keypoints that have low confidence score (below threshold), it will be placed by `-1`.

.. |keypoint_scores| replace:: ``keypoints_scores`` (:obj:`numpy.array`): a numpy array (N, keypoints, 1) containing
   the confidence scores of detected pose. N represents the number of detected poses, and K represents indiviudal
   keypoints.

.. |keypoints_conns| replace:: ``keypoints_conns`` (:obj:`numpy.array`): a numpy array containing the coordinates
   of keypoint connections between adjacent keypoints pairs, if both keypoints are detected.

.. |pipeline_end| replace:: ``pipeline_end`` (:obj:`bool`): A boolean that evaluates as ``True`` when
   the pipeline is completed. Suitable for operations that require the complete inference pipeline to
   be completed before running.

.. |filename| replace:: ``filename`` (:obj:`str`): the filename of video/image being read.

.. |fps| replace :: ``fps`` (:obj:`float`): the FPS of recorded video, upon filming.

.. |obj_3D_locs| replace:: ``obj_3D_locs`` (:obj:`list`): a list of numpy arrays (x,y,z) representing
   the 3D coordinates of an object associated with a detected bounding box.

.. |obj_groups| replace:: ``obj_groups`` (:obj:`list`): a list of integers, representing the assigned
   group number of an object associated with a detected bounding box.

.. |obj_tags| replace:: ``obj_tags`` (:obj:`list`): a list of strings to be added to a bounding box,
   for display. The order of the tags follow the order of "bboxes".

.. |zones| replace:: ``zones`` (:obj:`list`): A nested list of coordinates, with each sub-list containing
   the (x,y) coordinates that represent the points that form the boundaries of a zone. The order of zones
   follows the order of ``zone_counts``.

.. |zone_count| replace:: ``zone_count`` (:obj:`list`): A list of integers that represent the count of
   a pre-selected object (for example, people) detected in each specificed zone. The order of counts
   follows the order of ``zones``.


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
