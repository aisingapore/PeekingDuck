.. |img| replace:: ``img`` (:obj:`numpy.ndarray`): A numpy array (height, width, channels)
   representation of the image, in BGR format

.. |bboxes| replace:: ``bboxes`` (:obj:`numpy.ndarray`): A numpy array (N, 4) containing bounding
   box information of detected objects. N corresponds to the number of objects detected, and each
   bounding box is represented as (x1, y1, x2, y2), coordinates (x, y) of the top-left corner and
   bottom right corner of the bounding box respectively.

.. |bbox_labels| replace:: ``bbox_labels`` (:obj:`numpy.ndarray`): A numpy array (N) of strings,
   representing the labels of detected objects. The order corresponds to ``bboxes`` and
   ``bbox_scores``.

.. |bbox_scores| replace:: ``bbox_scores`` (:obj:`numpy.ndarray`): A numpy array (N) of the
   confidence scores [0, 1] of detected objects. The order corresponds to ``bboxes`` and
   ``bbox_labels``.

.. |btm_midpoint| replace:: ``btm_midpoint`` (:obj:`List[Tuple[int, int]]`): A list of tuples
   (x, y) each representing a single point of reference of bounding boxes for use in zone
   analytics. The order of ``btm_midpoint`` follows the order of ``bboxes``.

.. |count| replace:: ``count`` (:obj:`int`): An integer representing the number of counted objects.

.. |keypoints| replace:: ``keypoints`` (:obj:`numpy.ndarray`): A numpy array (N, K, 2) with the
   last dimension representing the coordinates (x, y) of detected poses. N represents the number
   of detected poses, and K represents individual keypoints. Keypoints with low confidence scores
   (below threshold) will be replaced by ``-1``.

.. |keypoint_scores| replace:: ``keypoint_scores`` (:obj:`numpy.ndarray`): A numpy array (N, K, 1)
   containing the confidence scores [0, 1] of detected poses. N represents the number of detected
   poses, and K represents individual keypoints.

.. |keypoint_conns| replace:: ``keypoint_conns`` (:obj:`List[numpy.ndarray]`): A list of N numpy
   arrays (2, 2) with each array representing one connection in the image. The numpy array contains
   the coordinates (x, y) of 2 adjacent keypoints pairs, if both keypoints are detected.

.. |pipeline_end| replace:: ``pipeline_end`` (:obj:`bool`): A boolean that evaluates as ``True``
   when the pipeline is completed. Suitable for operations that require the complete inference
   pipeline to be completed before running.

.. |filename| replace:: ``filename`` (:obj:`str`): The filename of video/image being read.

.. |fps| replace:: ``fps`` (:obj:`List[float]`): A list of floats representing the FPS per frame.
   The FPS returned can either be a moving average or an instantaneous value. This setting can be
   changed in the *configs/dabble/fps* file.

.. |saved_video_fps| replace:: ``saved_video_fps`` (:obj:`float`): FPS of the recorded video, upon
   filming.

.. |obj_3D_locs| replace:: ``obj_3D_locs`` (:obj:`List[numpy.ndarray]`): A list of N numpy arrays
   representing the 3D coordinates (x, y, z) of an object associated with a detected bounding box.

.. |obj_groups| replace:: ``obj_groups`` (:obj:`List[int]`): A list of integers representing the
   assigned group number of an object associated with a detected bounding box.

.. |large_groups| replace:: ``large_groups`` (:obj:`List[int]`): A list of integers representing
   the group IDs of groups that have exceeded the size threshold.

.. |obj_tags| replace:: ``obj_tags`` (:obj:`List[str]`): A list of strings to be added to a
   bounding box for display. The order of the tags follow the order of ``bboxes``.

.. |zones| replace:: ``zones`` (:obj:`List[List[Tuple[float, ...]]]`): A nested list of
   coordinates, with each sub-list containing the coordinates (x, y) representing the points that
   form the boundaries of a zone. The order of zones follows the order of ``zone_count``.

.. |zone_count| replace:: ``zone_count`` (:obj:`List[int]`): A list of integers representing the
   count of a pre-selected object (for example, "person") detected in each specified zone. The
   order of counts follows the order of ``zones``.

.. |density_map| replace:: ``density_map`` (:obj:`numpy.ndarray`): A numpy array that represents
   the number of persons per pixel. The sum of the array returns the total estimated count of people.

.. |none| replace:: ``none``: No inputs required, or no additional outputs produced.
   Used for ``input`` nodes that require no prior inputs, or ``draw`` nodes that overwrite current
   input.

.. |br| raw:: html

   <br />

.. |tab| unicode:: 0xA0 0xA0 0xA0 0xA0
   :trim:

.. |times|  unicode:: U+000D7 .. MULTIPLICATION SIGN

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
