..
   Data type substitutions

.. |all_input| replace:: ``all`` (:obj:`Any`): Receives inputs from all preceding outputs. For example, 
   in :mod:`draw.legend`, this is used as dynamic input for legend creation. In 
   :mod:`output.csv_writer`, this is used as flexible input for statistics to track.

.. |avg| replace:: ``avg`` (:obj:`float`): Average of an attribute over time.

.. |bboxes| replace:: ``bboxes`` (:obj:`numpy.ndarray`): A NumPy array of shape :math:`(N, 4)`
   containing normalized bounding box coordinates of detected objects where :math:`N` is the number
   of detected objects. Each bounding box is represented as :math:`(x_1, y_1, x_2, y_2)` where
   :math:`(x_1, y_1)` is the top left corner and :math:`(x_2, y_2)` is the bottom right corner.

.. |bbox_labels| replace:: ``bbox_labels`` (:obj:`numpy.ndarray`): A NumPy array of shape
   :math:`(N)` containing strings representing the labels of detected objects. The order
   corresponds to ``bboxes`` and ``bbox_scores``.

.. |bbox_scores| replace:: ``bbox_scores`` (:obj:`numpy.ndarray`): A NumPy array of shape
   :math:`(N)` containing confidence scores :math:`[0, 1]` of detected objects. The order
   corresponds to ``bboxes`` and ``bbox_labels``.

.. |btm_midpoint| replace:: ``btm_midpoint`` (:obj:`List[Tuple[int, int]]`): A list of tuples
   each representing the :math:`(x, y)` coordinate of the bottom middle of a bounding box for use
   in zone analytics. The order of corresponds to ``bboxes``.

.. |count| replace:: ``count`` (:obj:`int`): An integer representing the number of counted objects.

.. |density_map| replace:: ``density_map`` (:obj:`numpy.ndarray`): A NumPy array representing the
   number of persons per pixel. The sum of the array returns the total estimated count of people.

.. |filename| replace:: ``filename`` (:obj:`str`): The filename of video/image being read.

.. |fps| replace:: ``fps`` (:obj:`List[float]`): A list of floats representing the frames per
   second (FPS) per frame. The FPS returned can either be a moving average or an instantaneous
   value. This setting can be changed in the *configs/dabble/fps.yml* file.

.. |img| replace:: ``img`` (:obj:`numpy.ndarray`): A NumPy array of shape
   :math:`(height, width, channels)` containing the image data in BGR format.

.. |keypoints| replace:: ``keypoints`` (:obj:`numpy.ndarray`): A NumPy array of shape
   :math:`(N, K, 2)` containing the `x, y` coordinates of detected poses where :math:`N` is the
   number of detected poses, and :math:`K` is the number of individual keypoints. Keypoints with
   low confidence scores (below threshold) will be replaced by ``-1``.

.. |keypoint_conns| replace:: ``keypoint_conns`` (:obj:`numpy.ndarray`): A NumPy array of shape
   :math:`(N, D', 2)` containing the `x, y` coordinates of adjacent keypoint pairs. :math:`D'` is
   the number of valid keypoint pairs where both keypoints are detected.

.. |keypoint_scores| replace:: ``keypoint_scores`` (:obj:`numpy.ndarray`): A NumPy array of shape
   :math:`(N, K, 1)` containing the confidence scores of detected poses where :math:`N` is the
   number of detected poses and :math:`K` is the number of individual keypoints. The confidence
   score has a range of :math:`[0, 1]`.

.. |large_groups| replace:: ``large_groups`` (:obj:`List[int]`): A list of integers representing
   the group IDs of groups that have exceeded the size threshold.

.. |max| replace:: ``max`` (:obj:`float | int`): Maximum value of an attribute over time.

.. |min| replace:: ``min`` (:obj:`float | int`): Minimum value of an attribute over time.

.. |no_input| replace:: ``none``: No inputs required.

.. |no_output| replace:: ``none``: No outputs produced.

.. |obj_3D_locs| replace:: ``obj_3D_locs`` (:obj:`List[numpy.ndarray]`): A list of :math:`N` NumPy
   arrays representing the 3D coordinates :math:`(x, y, z)` of an object associated with a detected
   bounding box.

.. |obj_attrs| replace:: ``obj_attrs`` (:obj:`Dict[str, Any]`): A dictionary of attributes
   associated with each bounding box, in the same order as ``bboxes``. Different nodes that 
   produce this ``obj_attrs`` output type may contribute different attributes. For example, 
   ``dabble.tracking`` produces the ``ids`` attribute while ``dabble.check_nearby_objs`` produces 
   the ``flags`` attribute.

.. |pipeline_end| replace:: ``pipeline_end`` (:obj:`bool`): A boolean that evaluates to ``True``
   when the pipeline is completed. Suitable for operations that require the entire inference
   pipeline to be completed before running.

.. |saved_video_fps| replace:: ``saved_video_fps`` (:obj:`float`): FPS of the recorded video, upon
   filming.

.. |zones| replace:: ``zones`` (:obj:`List[List[Tuple[float, ...]]]`): A nested list of
   coordinates, with each sub-list containing the :math:`(x, y)` coordinates representing the points that
   form the boundaries of a zone. The order corresponds to ``zone_count``.

.. |zone_count| replace:: ``zone_count`` (:obj:`List[int]`): A list of integers representing the
   count of a pre-selected object class (for example, "person") detected in each specified zone.
   The order corresponds to ``zones``.


..
   Utility substitutions

.. |br| raw:: html

   <br />

.. |tab| unicode:: 0xA0 0xA0 0xA0 0xA0
   :trim:

.. |times| unicode:: U+000D7 .. MULTIPLICATION SIGN
