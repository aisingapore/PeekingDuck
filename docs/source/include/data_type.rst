..
   Data type substitutions

..
   Docstring config substitutions

.. |all_input_data| replace:: |all|: |all_input_def|

.. |bboxes_data| replace:: |bboxes|: |bboxes_def|

.. |bbox_labels_data| replace:: |bbox_labels|: |bbox_labels_def|

.. |bbox_scores_data| replace:: |bbox_scores|: |bbox_scores_def|

.. |btm_midpoint_data| replace:: |btm_midpoint|: |btm_midpoint_def|

.. |count_data| replace:: |count|: |count_def|

.. |cum_avg_data| replace:: |cum_avg|: |cum_avg_def|

.. |cum_max_data| replace:: |cum_max|: |cum_max_def|

.. |cum_min_data| replace:: |cum_min|: |cum_min_def|

.. |density_map_data| replace:: |density_map|: |density_map_def|

.. |filename_data| replace:: |filename|: |filename_def|

.. |fps_data| replace:: |fps|: |fps_def|

.. |img_data| replace:: |img|: |img_def|

.. |keypoints_data| replace:: |keypoints|: |keypoints_def|

.. |keypoint_conns_data| replace:: |keypoint_conns|: |keypoint_conns_def|

.. |keypoint_scores_data| replace:: |keypoint_scores|: |keypoint_scores_def|

.. |large_groups_data| replace:: |large_groups|: |large_groups_def|

.. |none_input_data| replace:: |none|: |none_input_def|

.. |none_output_data| replace:: |none|: |none_output_def|

.. |obj_3D_locs_data| replace:: |obj_3D_locs|: |obj_3D_locs_def|

.. |obj_attrs_data| replace:: |obj_attrs|: |obj_attrs_def|

.. |pipeline_end_data| replace:: |pipeline_end|: |pipeline_end_def|

.. |saved_video_fps_data| replace:: |saved_video_fps|: |saved_video_fps_def|

.. |zones_data| replace:: |zones|: |zones_def|

.. |zone_count_data| replace:: |zone_count|: |zone_count_def|

..
   Glossary term/reference substitutions

.. |all| replace:: ``all`` (:obj:`Any`)

.. |bboxes| replace:: ``bboxes`` (:obj:`numpy.ndarray`)

.. |bbox_labels| replace:: ``bbox_labels`` (:obj:`numpy.ndarray`)

.. |bbox_scores| replace:: ``bbox_scores`` (:obj:`numpy.ndarray`)

.. |btm_midpoint| replace:: ``btm_midpoint`` (:obj:`List[Tuple[int, int]]`)
   
.. |count| replace:: ``count`` (:obj:`int`)
   
.. |cum_avg| replace:: ``cum_avg`` (:obj:`float`)
   
.. |cum_max| replace:: ``cum_max`` (:obj:`float | int`)
   
.. |cum_min| replace:: ``cum_min`` (:obj:`float | int`)
   
.. |density_map| replace:: ``density_map`` (:obj:`numpy.ndarray`)

.. |filename| replace:: ``filename`` (:obj:`str`)
   
.. |fps| replace:: ``fps`` (:obj:`float`)
   
.. |img| replace:: ``img`` (:obj:`numpy.ndarray`)
   
.. |keypoints| replace:: ``keypoints`` (:obj:`numpy.ndarray`)
   
.. |keypoint_conns| replace:: ``keypoint_conns`` (:obj:`numpy.ndarray`)
   
.. |keypoint_scores| replace:: ``keypoint_scores`` (:obj:`numpy.ndarray`)

.. |large_groups| replace:: ``large_groups`` (:obj:`List[int]`)
   
.. |none| replace:: ``none``
   
.. |obj_3D_locs| replace:: ``obj_3D_locs`` (:obj:`List[numpy.ndarray]`)
   
.. |obj_attrs| replace:: ``obj_attrs`` (:obj:`Dict[str, Any]`)

.. |pipeline_end| replace:: ``pipeline_end`` (:obj:`bool`)
   
.. |saved_video_fps| replace:: ``saved_video_fps`` (:obj:`float`)
   
.. |zones| replace:: ``zones`` (:obj:`List[List[Tuple[float, ...]]]`)
   
.. |zone_count| replace:: ``zone_count`` (:obj:`List[int]`)

..
   Glossary definition substitutions

.. |all_input_def| replace:: This data type contains all the outputs from
   preceding nodes, granting a large degree of flexibility to nodes that receive
   it. Examples of such nodes include :mod:`draw.legend`,
   :mod:`dabble.statistics`, and :mod:`output.csv_writer`.

.. |bboxes_def| replace:: A NumPy array of shape :math:`(N, 4)` containing
   normalized bounding box coordinates of :math:`N` detected objects. Each
   bounding box is represented as :math:`(x_1, y_1, x_2, y_2)` where
   :math:`(x_1, y_1)` is the top-left corner and :math:`(x_2, y_2)` is the
   bottom right corner. The order corresponds to :term:`bbox_labels` and
   :term:`bbox_scores`.

.. |bbox_labels_def| replace:: A NumPy array of shape :math:`(N)` containing
   strings representing the labels of detected objects. The order corresponds to
   :term:`bboxes` and :term:`bbox_scores`.

.. |bbox_scores_def| replace:: A NumPy array of shape :math:`(N)` containing
   confidence scores :math:`[0, 1]` of detected objects. The order corresponds
   to :term:`bboxes` and :term:`bbox_labels`.

.. |btm_midpoint_def| replace:: A list of tuples each representing the
   :math:`(x, y)` coordinates of the bottom middle of a bounding box for use in
   zone analytics. The order corresponds to :term:`bboxes`.

.. |count_def| replace:: An integer representing the number of counted objects.

.. |cum_avg_def| replace:: Cumulative average of an attribute over time.

.. |cum_max_def| replace:: Cumulative maximum of an attribute over time.

.. |cum_min_def| replace:: Cumulative minimum of an attribute over time.

.. |density_map_def| replace:: A NumPy array of shape :math:`(H, W)`
   representing the number of persons per pixel. :math:`H` and :math:`W` are the
   height and width of the input image, respectively. The sum of the array
   is the estimated total number of people.

.. |filename_def| replace:: The filename of video/image being read.

.. |fps_def| replace:: A float representing the Frames Per Second (FPS) when
   processing a live video stream or a recorded video.

.. |img_def| replace:: A NumPy array of shape :math:`(height, width, channels)`
   containing the image data in BGR format.

.. |keypoints_def| replace:: A NumPy array of shape :math:`(N, K, 2)` containing
   the :math:`(x, y)` coordinates of detected poses where :math:`N` is the
   number of detected poses, and :math:`K` is the number of individual
   keypoints. Keypoints with low confidence scores (below threshold) will be
   replaced by ``-1``.

.. |keypoint_conns_def| replace:: A NumPy array of shape :math:`(N, D_n', 2, 2)`
   containing the :math:`(x, y)` coordinates of adjacent keypoint pairs where
   :math:`N` is the number of detected poses, and :math:`D_n'` is the number of
   valid keypoint pairs for the the :math:`n`-th pose where both keypoints are
   detected.

.. |keypoint_scores_def| replace:: A NumPy array of shape :math:`(N, K)`
   containing the confidence scores of detected poses where :math:`N` is the
   number of detected poses and :math:`K` is the number of individual keypoints.
   The confidence score has a range of :math:`[0, 1]`.

.. |large_groups_def| replace:: A list of integers representing the group IDs of
   groups that have exceeded the size threshold.

.. |none_input_def| replace:: No inputs required.

.. |none_output_def| replace:: No outputs produced.

.. |obj_3D_locs_def| replace:: A list of :math:`N` NumPy arrays representing the
   3D coordinates :math:`(x, y, z)` of an object associated with a detected
   bounding box.

.. |obj_attrs_def| replace:: A dictionary of attributes associated with each
   bounding box, in the same order as :term:`bboxes`. Different nodes that
   produce this :term:`obj_attrs` output type may contribute different
   attributes.

.. |pipeline_end_def| replace:: A boolean that evaluates to ``True`` when the
   pipeline is completed. Suitable for operations that require the entire
   inference pipeline to be completed before running.

.. |saved_video_fps_def| replace:: FPS of the recorded video, upon filming.

.. |zones_def| replace:: A nested list of :math:`Z` zones. Each zone is
   described by :math:`3` **or more** points which contains the :math:`(x, y)`
   coordinates forming the boundary of a zone. The order corresponds to
   :term:`zone_count`.

.. |zone_count_def| replace:: A list of integers representing the count of a
   pre-selected object class (for example, "person") detected in each specified
   zone. The order corresponds to :term:`zones`.