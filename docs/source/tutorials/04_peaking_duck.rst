************
Peaking Duck
************

.. include:: /include/substitution.rst

PeekingDuck includes some "power" nodes that are capable of processing the contents 
or outputs of the other nodes and to accumulate information over time.
An example is the :mod:`dabble.statistics` node which can accumulate statistical 
information, such as calculating the cumulative average and maximum of particular 
objects (like people or cars).
This tutorial presents advanced recipes to showcase the power features of 
PeekingDuck, such as using :mod:`dabble.statistics` for object counting and tracking.


.. _tutorial_sql:

Interfacing with SQL
====================

This tutorial demonstrates how to save data to an SQLite database.
We will extend the tutorial for :ref:`counting hand waves<tutorial_count_hand_wave>`
with a new custom :mod:`output` node that writes information into a local SQLite database.

   .. note::

      The above tutorial assumes ``sqlite3`` has been installed in your system. |br|
      If your system does not have ``sqlite3``, please see the `SQLite Home Page 
      <http://www.sqlite.org/>`_ for installation instructions.

First, create a new custom ``output.sqlite`` node in the ``custom_project`` folder:

   .. admonition:: Terminal Session

      | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck create-node` \
      | Creating new custom node...
      | Enter node directory relative to ~user/wave_project [src/custom_nodes]: \ :green:`⏎` \
      | Select node type (input, augment, model, draw, dabble, output): \ :green:`output` \
      | Enter node name [my_custom_node]: \ :green:`sqlite` \
      |
      | Node directory:	~user/wave_project/src/custom_nodes
      | Node type:	output
      | Node name:	sqlite
      |
      | Creating the following files:
      |    Config file: ~user/wave_project/src/custom_nodes/configs/output/sqlite.yml
      |    Script file: ~user/wave_project/src/custom_nodes/output/sqlite.py
      | Proceed? [Y/n]: \ :green:`⏎` \
      | Created node!

The updated folder structure would be:

   .. parsed-literal::

      \ :blue:`wave_project/` \ |Blank|
      ├── pipeline_config.yml
      ├── \ :blue:`src/` \ |Blank|
      │   └── \ :blue:`custom_nodes/` \ |Blank|
      │       ├── \ :blue:`configs/` \ |Blank|
      │       │   ├── \ :blue:`dabble/` \ |Blank|
      │       │   │   └── wave.yml
      │       │   └── \ :blue:`output/` \ |Blank|
      │       │       └── sqlite.yml
      │       ├── \ :blue:`dabble/` \ |Blank|
      │       │   └── wave.py
      │       └── \ :blue:`output/` \ |Blank|
      │           └── sqlite.py
      └── wave.mp4


Edit the following **five files** as described below:

#. **src/custom_nodes/configs/output/sqlite.yml**:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["hand_direction", "num_waves"]
      output: ["none"]

      # No optional configs

   The new ``output.sqlite`` custom node will take in the hand direction and the
   current number of hand waves to save to the external database.


#. **src/custom_nodes/output/sqlite.py**:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for sqlite.py**

      .. code-block:: python
         :linenos:
   
         """
         Custom node to save data to external database.
         """
   
         from typing import Any, Dict
         from datetime import datetime
         from peekingduck.nodes.abstract_node import AbstractNode
         import sqlite3
   
         DB_FILE = "wave.db"           # name of database file
   
   
         class Node(AbstractNode):
            """Custom node to save hand direction and current wave count to database.
   
            Args:
               config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
            """
   
            def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
               super().__init__(config, node_path=__name__, **kwargs)
   
               self.conn = None
               try:
                  # try to establish connection to database,
                  # will create DB_FILE if it does not exist
                  self.conn = sqlite3.connect(DB_FILE)
                  self.logger.info(f"Connected to {DB_FILE}")
                  sql = """ CREATE TABLE IF NOT EXISTS wavetable (
                                 datetime text,
                                 hand_direction text,
                                 wave_count integer
                            ); """
                  cur = self.conn.cursor()
                  cur.execute(sql)
               except sqlite3.Error as e:
                  self.logger.info(f"SQL Error: {e}")
   
            def update_db(self, hand_direction: str, num_waves: int) -> None:
               """Helper function to save current time stamp, hand direction and 
               wave count into DB wavetable.
               """
               now = datetime.now()
               dt_str = f"{now:%Y-%m-%d %H:%M:%S}"
               sql = """ INSERT INTO wavetable(datetime,hand_direction,wave_count) 
                         values (?,?,?) """
               cur = self.conn.cursor()
               cur.execute(sql, (dt_str, hand_direction, num_waves))
               self.conn.commit()
   
            def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
               """Node to output hand wave data into sqlite database.
   
               Args:
                     inputs (dict): Dictionary with keys "hand_direction", "num_waves"
   
               Returns:
                     outputs (dict): Empty dictionary
               """
   
               hand_direction = inputs["hand_direction"]
               num_waves = inputs["num_waves"]
               self.update_db(hand_direction, num_waves)
   
               return {}

   This tutorial uses the ``sqlite3`` package to interface with the database.

   On first run, the node initializer will create the ``wave.db`` database file.
   It will establish a connection to the database and create a table called
   ``wavetable`` if it does not exist.
   This table is used to store the hand direction and wave count data.
   
   A helper function ``update_db`` is called to update the database.
   It saves the current date time stamp, hand direction and wave count into the 
   ``wavetable``.
   
   The node's ``run`` method retrieves the required inputs from the pipeline's 
   data pool and calls ``self.update_db`` to save the data.


#. **src/custom_nodes/configs/dabble/wave.yml**:

   .. code-block:: yaml
      :linenos:

      # Dabble node has both input and output
      input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]
      output: ["hand_direction", "num_waves"]

      # No optional configs

   To support the ``output.sqlite`` custom node's input requirements, we need to 
   modify the ``dabble.wave`` custom node to return the current hand direction
   ``hand_direction`` and the current wave count ``num_waves``.


#. **src/custom_nodes/dabble/wave.py**:

   .. code-block:: python
      :lineno-start: 173

         ... same as previous ...
         return {
            "hand_direction": self.direction if self.direction is not None else "None",
            "num_waves": self.num_waves,
         }

   This file is the same as the ``wave.py`` in the :ref:`counting hand
   waves<tutorial_count_hand_wave>` tutorial, except for the changes in the last few
   lines as shown above.
   These changes outputs the ``hand_direction`` and ``num_waves`` to the pipeline's 
   data pool for subsequent consumption by the ``output.sqlite`` custom node.


#. **pipeline_config.yml**:

   .. code-block:: yaml
      :lineno-start: 11

      ... same as previous ...
      - custom_nodes.output.sqlite

   Likewise, the pipeline is the same as in the previous tutorial, except for 
   line 12 that has been added to call the new custom node.
   
Run this project with :greenbox:`peekingduck run` and when completed, a new ``wave.db`` 
sqlite database file would be created in the current folder.
Examine the created database as follows:

   .. admonition:: Terminal Session

      | \ :blue:`[~user/wave_project]` \ > \ :green:`sqlite3` \
      | SQLite version 3.37.0 2021-11-27 14:13:22
      | Enter ".help" for usage hints.
      | Connected to a transient in-memory database.
      | Use ".open FILENAME" to reopen on a persistent database.
      | sqlite> \ :green:`.open wave.db` \
      | sqlite> \ :green:`.schema wavetable` \
      | CREATE TABLE wavetable (
      |                             datetime text,
      |                             hand_direction text,
      |                             wave_count integer
      |                         );
      | sqlite> \ :green:`select * from wavetable where wave_count > 0 limit 5;` \
      | 2022-02-15 19:26:16|left|1
      | 2022-02-15 19:26:16|right|1
      | 2022-02-15 19:26:16|left|2
      | 2022-02-15 19:26:16|right|2
      | 2022-02-15 19:26:16|right|2
      | sqlite> \ :green:`select * from wavetable order by datetime desc limit 5;` \
      | 2022-02-15 19:26:44|right|72
      | 2022-02-15 19:26:44|right|72
      | 2022-02-15 19:26:44|right|72
      | 2022-02-15 19:26:44|right|72
      | 2022-02-15 19:26:43|right|70

Press :greenbox:`CTRL-D` to exit from ``sqlite3``.


.. _tutorial_counting_cars:

Counting Cars
=============

This tutorial demonstrates using the :mod:`dabble.statistics` node to count the number
of cars traveling across a highway over time and the :mod:`draw.legend` node to display
the relevant statistics.

Create a new PeekingDuck project, download the `highway cars video
<https://storage.googleapis.com/peekingduck/videos/highway_cars.mp4>`_ and save it into the project
folder.

   .. admonition:: Terminal Session

      | \ :blue:`[~user]` \ > \ :green:`mkdir car_project` \
      | \ :blue:`[~user]` \ > \ :green:`cd car_project` \
      | \ :blue:`[~user/car_project]` \ > \ :green:`peekingduck init` \

The ``car_project`` folder structure:

   .. parsed-literal::

      \ :blue:`car_project/` \ |Blank|
      ├── highway_cars.mp4
      ├── pipeline_config.yml
      └── \ :blue:`src` \ |Blank|
         └── custom_nodes
            └── configs


Edit ``pipeline_config.yml`` as follows:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.visual:
          source: highway_cars.mp4
      - model.yolo:
          detect: ["car"]
          score_threshold: 0.3
      - dabble.bbox_count
      - dabble.fps
      - dabble.statistics:
          identity: count
      - draw.bbox
      - draw.legend:
          show: ["fps", "count", "cum_max", "cum_min"]
      - output.screen

Run it with :greenbox:`peekingduck run` and you should see a video of cars travelling
across a highway with a legend box on the bottom left showing the realtime count of the
number of cars on-screen, the cumulative maximum and minimum number of cars detected
since the video started.
The sample screenshot below shows:

   * the count that there are currently 3 cars on-screen
   * the cumulative maximum number of cars "seen" previously was 5
   * the cumulative minimum number of cars was 1

   .. figure:: /assets/tutorials/ss_highway_cars.png
      :width: 394
      :alt: PeekingDuck screenshot - counting cars

      Counting Cars on a Highway

   .. note::

      Royalty free video of cars on highway from: 
      https://www.youtube.com/watch?v=8yP1gjg4b2w


.. _tutorial_object_tracking:

Object Tracking
===============

Object tracking is the application of CV models to automatically detect objects 
in a video and to assign a unique identity to each of them.
These objects can be either living (e.g. person) or non-living (e.g. car). 
As they move around in the video, these objects are identified based on 
their assigned identities and tracked according to their movements.

This tutorial demonstrates using :mod:`dabble.statistics` with a custom node to 
track the number of people walking down a path.

Create a new PeekingDuck project, download the `people walking video
<https://storage.googleapis.com/peekingduck/videos/people_walking.mp4>`_ and save it into the project
folder.

   .. admonition:: Terminal Session

      | \ :blue:`[~user]` \ > \ :green:`mkdir people_walking` \
      | \ :blue:`[~user]` \ > \ :green:`cd people_walking` \
      | \ :blue:`[~user/people_walking]` \ > \ :green:`peekingduck init` \

Create the following ``pipeline_config.yml``:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.visual:
          source: people_walking.mp4
      - model.yolo:
          detect: ["person"]
      - dabble.tracking
      - dabble.statistics:
          maximum: obj_attrs["ids"]
      - dabble.fps
      - draw.bbox
      - draw.tag:
          show: ["ids"]
      - draw.legend:
          show: ["fps", "cum_max", "cum_min", "cum_avg"]
      - output.screen

The above pipeline uses the YOLO model to detect people in the video and uses 
the :mod:`dabble.tracking` node to track the people as they walk.
Each person is assigned a tracking ID and :mod:`dabble.tracking` returns a list of 
tracking IDs.
:mod:`dabble.statistics` is used to process these tracking IDs: since each person is 
assigned a monotonically increasing integer ID, the maximum ID within the list 
tells us the number of persons tracked so far.
:mod:`draw.tag` shows the ID above the tracked person.
:mod:`draw.legend` is used to display the various statistics: the FPS, and the 
cumulative maximum, minimum and average relating to the number of persons tracked.

Do a :greenbox:`peekingduck run` and you will see the following display:

   .. figure:: /assets/tutorials/ss_people_walking_1.png
      :width: 394
      :alt: PeekingDuck screenshot - people walking

      People Walking

   .. note::

      Royalty free video of people walking from:
      https://www.youtube.com/watch?v=du74nvmRUzo

.. _tutorial_tracking_within_zone:

.. _tutorial_tracking_people_within_zone:

Tracking People within a Zone
-----------------------------

Suppose we are only interested in people walking down the center of the video 
(imagine a carpet running down the middle).
We can create a custom node to tell PeekingDuck to focus on the middle zone, 
by filtering away the detected bounding boxes outside the zone.

Start by creating a custom node ``dabble.filter_bbox``:

   .. admonition:: Terminal Session

      | \ :blue:`[~user/people_walking]` \ > \ :green:`peekingduck create-node` \
      | Creating new custom node...
      | Enter node directory relative to ~user/people_walking [src/custom_nodes]: \ :green:`⏎` \
      | Select node type (input, augment, model, draw, dabble, output): \ :green:`dabble` \
      | Enter node name [my_custom_node]: \ :green:`filter_bbox` \
      |
      | Node directory:	~user/people_walking/src/custom_nodes
      | Node type:	dabble
      | Node name:	filter_bbox
      |
      | Creating the following files:
      |    Config file: ~user/people_walking/src/custom_nodes/configs/dabble/filter_bbox.yml
      |    Script file: ~user/people_walking/src/custom_nodes/dabble/filter_bbox.py
      | Proceed? [Y/n]: \ :green:`⏎` \
      | Created node!

The folder structure looks like this:

   .. parsed-literal::

      \ :blue:`people_walking/` \ |Blank|
      ├── people_walking.mp4
      ├── pipeline_config.yml
      └── src
         └── custom_nodes
            ├── configs
            │   └── dabble
            │       └── filter_bbox.yml
            └── dabble
                  └── filter_bbox.py


Change ``pipeline_config.yml`` to the following:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.visual:
          source: people_walking.mp4
      - model.yolo:
          detect: ["person"]
      - dabble.bbox_to_btm_midpoint
      - dabble.zone_count:
          resolution: [720, 480]
          zones: [
            [[0.35,0], [0.65,0], [0.65,1], [0.35,1]],
          ]
      - custom_nodes.dabble.filter_bbox:
          zones: [
            [[0.35,0], [0.65,0], [0.65,1], [0.35,1]],
          ]
      - dabble.tracking
      - dabble.statistics:
          maximum: obj_attrs["ids"]
      - dabble.fps
      - draw.bbox
      - draw.zones
      - draw.tag:
          show: ["ids"]
      - draw.legend:
          show: ["fps", "cum_max", "cum_min", "cum_avg", "zone_count"]
      - output.screen

We make use of :mod:`dabble.zone_count` and :mod:`dabble.bbox_to_btm_midpoint` nodes to 
create a zone in the middle. The zone is defined by a rectangle with the 
four corners (0.35, 0.0) - (0.65, 0.0) - (0.65, 1.0) - (0.35, 1.0).
(For more info, see :doc:`Zone Counting </use_cases/zone_counting>`)
This zone is also defined in our custom node ``dabble.filter_bbox`` for bounding box
filtering.
What ``dabble.filter_bbox`` will do is to take the list of bboxes as input and 
output a list of bboxes within the zone, dropping all bboxes outside it.
Then, :mod:`dabble.tracking` is used to track the people walking and 
:mod:`dabble.statistics` is used to determine the number of people walking in the zone,
by getting the maximum of the tracked IDs.
:mod:`draw.legend` has a new item :term:`zone_count` which displays the number of people 
walking in the zone currently.

The ``filter_bbox.yml`` and ``filter_bbox.py`` files are shown below:

#. **src/custom_nodes/configs/dabble/filter_bbox.yml**:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["bboxes"]
      output: ["bboxes"]

      zones: [
         [[0,0], [0,1], [1,1], [1,0]],
      ]

   .. note::

      The ``zones`` default value of ``[[0,0], [0,1], [1,1], [1,0]]`` will be overridden
      by those specified in ``pipeline_config.yml`` above.
      See :ref:`Configuration - Behind The Scenes<tutorial_behind_the_scenes>` for more
      details.

#. **src/custom_nodes/dabble/filter_bbox.py**:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for filter_bbox.py**

      .. code-block:: python
         :linenos:

         """
         Custom node to filter bboxes outside a zone
         """

         from typing import Any, Dict
         import numpy as np
         from peekingduck.nodes.abstract_node import AbstractNode


         class Node(AbstractNode):
            """Custom node to filter bboxes outside a zone

            Args:
               config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
            """

            def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
               super().__init__(config, node_path=__name__, **kwargs)

            def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
               """Checks bounding box x-coordinates against the zone left and right borders.
               Retain bounding box if within, otherwise discard it.

               Args:
                     inputs (dict): Dictionary with keys "bboxes"

               Returns:
                     outputs (dict): Dictionary with keys "bboxes".
               """
               bboxes = inputs["bboxes"]
               zones = self.config["zones"]
               zone = zones[0]         # only work with one zone currently
               # convert zone with 4 points to a zone bbox with (x1, y1), (x2, y2)
               x1, y1 = zone[0]
               x2, y2 = zone[2]
               zone_bbox = np.asarray([x1, y1, x2, y2])

               retained_bboxes = []
               for bbox in bboxes:
                  # filter by left and right borders (ignore top and bottom)
                  if bbox[0] > zone_bbox[0] and bbox[2] < zone_bbox[2]:
                     retained_bboxes.append(bbox)

               return {"bboxes": np.asarray(retained_bboxes)}


Do a :greenbox:`peekingduck run` and you will see the following display:

   .. figure:: /assets/tutorials/ss_people_walking_2.png
      :width: 394
      :alt: PeekingDuck screenshot - count people walking in a zone

      Count People Walking in a Zone


.. _tutorial_callbacks:

Using Callbacks
===============

While PeekingDuck allows you to extend its capabilities through custom nodes, it
may not be the most optimal/efficient approach in some cases. For example, if you
are trying to bring in PeekingDuck into an existing codebase/project, it may not
be desirable to refactor the existing codebase such that it can be constructed/initialized
in a custom node.


Introduction to Using Callbacks in PeekingDuck
----------------------------------------------

Callbacks are externally defined functions/methods which are invoked as specific
points of a PeekingDuck pipeline and can interact with code/systems outside of PeekingDuck.
They can be registered to nodes in a pipeline using the ``callbacks`` config option.

Pipeline Events
^^^^^^^^^^^^^^^

The following pipeline events and their respective event keys are implemented:

   #. ``run_begin``: Invoked before a ``Node``'s ``run()`` method.
   #. ``run_end``: Invoked after a ``Nodes``'s ``run()`` method.

For example, in the following pipeline config:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.visual
      - model.posenet:
          callbacks:
            run_begin: [<callback 1>]
            run_end: [<callback 2>, <callback 3>, <callback 4>]
      - draw.poses
      - output.screen

After a image frame is read by :mod:`input.visual`, ``<callback 1>`` is invoked,
followed by :mod:`model.posenet`'s ``run()`` method, then ``<callback 2>``, ``<callback 3>``,
and ``<callback 4>`` are invoked after that.

Callback Definition
^^^^^^^^^^^^^^^^^^^

Callback definition, i.e. ``<callback 1>`` and ``<callback 2>`` in the previous
example, follows a strict format. The following definition has been written using
the `EBNF notation <https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form>`_:

   .. parsed-literal::
      
      callback definition = module part, "::", callback part
      module part         = { subdir name, "." }, script name
      callback part       = function
                           | class name, "::", method
                           | instance name, "::", instance method
      
      script name     = ? name of callback Python script ?
      subdir name     = ? name of individual subdirectory folders relative to "callbacks" folder ?
      class name      = ? name of class which contains the callback methods ?
      instance name   = ? name of instantiated object which contains the callback method ?
      function        = ? name of callback function ?
      method          = ? name of class/static methods ?
      instance method = ? name of instance method ? 

When using ``<instance name>::<instance method>`` for ``callback part``, the
instance object has to be in the script's global scope. 

The ``callbacks`` directory, which houses the scripts containing the callback functions,
is expected to be at the same location of the pipeline config file. For example,
given the following directory structure and file contents:

   .. parsed-literal::

      \ :blue:`callbacks_project/`
      ├── pipeline_config.yml
      └── \ :blue:`callbacks/`
          ├── my_callback.py
          └── \ :blue:`cb_dir/`
              └── my_cb.py

#. **callbacks/my_callback.py**

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for my_callback.py**

      .. code-block:: python
         :linenos:

         def general_function(data_pool):
             print("Called general_function")
             filename = data_pool["filename"]
             print(f"  filename={filename}")


         class MyCallbackClass:
             @staticmethod
             def the_static_method(data_pool):
                 print("Called the_static_method")

             @classmethod
             def the_class_method(cls, data_pool):
                 print("Called the_class_method")

             def the_instance_method(self, data_pool):
                 print("Called the_instance_method")
                 kp_scores = data_pool["keypoint_scores"][0]
                 print(f"  num_keypoint_scores={len(kp_scores)}")


         # my_callback_obj has to be visible from my_callback.py's global scope
         # to enable access to the_instance_method()
         my_callback_obj = MyCallbackClass()


#. **callbacks/cb_dir/my_cb.py**

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for my_cb.py**

      .. code-block:: python
         :linenos:

         def my_cb_function(data_pool):
             print("Called my_cb_function")

The various callback functions and methods can be registered in ``pipeline_config.yml``
as follows:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.visual:
          source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
      - model.posenet:
          callbacks:
            run_begin:
              [
                "my_callback::MyCallbackClass::the_static_method",
                "my_callback::MyCallbackClass::the_class_method",
              ]
            run_end:
              [
                "my_callback::general_function",
                "my_callback::my_callback_obj::the_static_method",
                "my_callback::my_callback_obj::the_class_method",
                "my_callback::my_callback_obj::the_instance_method",
                "cb_dir.my_cb::my_cb_function",
              ]
      - draw.poses
      - output.screen

The following table illustrates how the callback definitions map to their respective
callback functions:

+---------------------------------------------------------------+----------------------------------------------------+---------------------------------------------+
| Definition                                                    | Function                                           | Script location                             |
+===============================================================+====================================================+=============================================+
| ``my_callback::general_function``                             | ``general_function()``                             | ``callbacks/my_callback.py``                |
+---------------------------------------------------------------+----------------------------------------------------+                                             +
| ``my_callback::MyCallbackClass::the_static_method``           | ``MyCallbackClass.the_static_method()``            |                                             |
+---------------------------------------------------------------+                                                    +                                             +
| ``my_callback::my_callback_obj::the_static_method``           |                                                    |                                             |
+---------------------------------------------------------------+----------------------------------------------------+                                             +
| ``my_callback::MyCallbackClass::the_class_method``            | ``MyCallbackClass.the_class_method()``             |                                             |
+---------------------------------------------------------------+                                                    +                                             +
| ``my_callback::my_callback_obj::the_class_method``            |                                                    |                                             |
+---------------------------------------------------------------+----------------------------------------------------+                                             +
| ``my_callback::my_callback_obj::the_instance_method``         | ``MyCallbackClass::the_instance_method()``         |                                             |
+---------------------------------------------------------------+----------------------------------------------------+---------------------------------------------+
| ``cb_dir.my_cb::my_cb_function``                              | ``my_cb_function()``                               | ``callbacks/cb_dir/my_cb.py``               |
+---------------------------------------------------------------+----------------------------------------------------+---------------------------------------------+

Running the pipeline with :greenbox:`peekingduck run` should give the following output:

   .. admonition:: Terminal Session

      | [Truncated]
      |
      | 2022-12-01 17:33:23 peekingduck.declarative_loader  INFO:  Config for node model.posenet is updated to: 'callbacks': {'run_begin': ['my_callback::MyCallbackClass::the_static_method', 'my_callback::MyCallbackClass::the_class_method'], 'run_end': ['my_callback::general_function', 'my_callback::my_callback_obj::the_static_method', 'my_callback::my_callback_obj::the_class_method', 'my_callback::my_callback_obj::the_instance_method', 'cb_dir.my_cb::my_cb_function']}
      | 2022-12-01 17:33:23 peekingduck.nodes.model.posenetv1.posenet_files.predictor  INFO:  PoseNet model loaded with following configs:
      |         Model type: resnet,
      |         Input resolution: (225, 225),
      |         Max pose detection: 10,
      |         Score threshold: 0.4
      | 2022-12-01 17:33:26 peekingduck.declarative_loader  INFO:  Initializing draw.poses node...
      | 2022-12-01 17:33:26 peekingduck.declarative_loader  INFO:  Initializing output.screen node...
      | Called the_static_method
      | Called the_class_method
      | Called general_function
      |   filename=video.mp4
      | Called the_static_method
      | Called the_class_method
      | Called the_instance_method
      |   num_keypoint_scores=17
      | Called my_cb_function
      |
      | [Truncated]


Interfacing with SQL Using Callbacks
------------------------------------

This tutorial replicates the earlier :ref:`Interfacing with SQL<tutorial_sql>` tutorial
but uses callbacks instead of custom nodes.

   .. note::

      The above tutorial assumes ``sqlite3`` has been installed in your system. |br|
      If your system does not have ``sqlite3``, please see the `SQLite Home Page 
      <http://www.sqlite.org/>`_ for installation instructions.

The following steps are necessary to prepare the files and folder structure required
for this tutorial:

   #. Create and initialize a new PeekingDuck project in a folder named
      ``callbacks_wave_project``.
   #. Create a ``dabble.wave`` custom node.
   #. Create a ``sql_callback.py`` in the ``callbacks/`` folder.
   #. Copy `wave.mp4 <https://storage.googleapis.com/peekingduck/videos/wave.mp4>`_
      into the folder.

You should end up with the following folder structure:

   .. parsed-literal::

      \ :blue:`callbacks_wave_project/`
      ├── pipeline_config.yml
      ├── \ :blue:`callbacks/`
      │   └── sql_callback.py
      ├── \ :blue:`src/`
      │   └── \ :blue:`custom_nodes/`
      │       ├── \ :blue:`configs/`
      │       │   └── \ :blue:`dabble/`
      │       │       └── wave.yml
      │       └── \ :blue:`dabble/`
      │           └── wave.py
      └── wave.mp4

Edit the following **files** as described below:

#. **src/custom_nodes/configs/dabble/wave.yml**:

   .. code-block:: yaml
      :linenos:

      # Dabble node has both input and output
      input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]
      output: ["hand_direction", "num_waves"]

      callbacks: {}

   The additional ``callbacks: {}`` config allows us to override it later in ``pipeline_config.yml``
   to define the callback functions to use.

#. **src/custom_nodes/dabble/wave.py**:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for wave.py**

      .. code-block:: python
         :linenos:

         """
         Custom node to show keypoints and count the number of times the person's hand is waved
         """

         from typing import Any, Dict, List, Tuple

         import cv2
         from peekingduck.pipeline.nodes.abstract_node import AbstractNode

         # setup global constants
         FONT = cv2.FONT_HERSHEY_SIMPLEX
         WHITE = (255, 255, 255)  # opencv loads file in BGR format
         YELLOW = (0, 255, 255)
         THRESHOLD = 0.6  # ignore keypoints below this threshold
         KP_RIGHT_SHOULDER = 6  # PoseNet's skeletal keypoints
         KP_RIGHT_WRIST = 10


         def map_bbox_to_image_coords(
             bbox: List[float], image_size: Tuple[int, int]
         ) -> Tuple[int, ...]:
             """First helper function to convert relative bounding box coordinates to
             absolute image coordinates.
             Bounding box coords ranges from 0 to 1
             where (0, 0) = image top-left, (1, 1) = image bottom-right.

             Args:
                 bbox (List[float]): List of 4 floats x1, y1, x2, y2
                 image_size (Tuple[int, int]): Width, Height of image

             Returns:
                 Tuple[int, ...]: x1, y1, x2, y2 in integer image coords
             """
             width, height = image_size[0], image_size[1]
             x1, y1, x2, y2 = bbox
             x1 *= width
             x2 *= width
             y1 *= height
             y2 *= height
             return int(x1), int(y1), int(x2), int(y2)

         def map_keypoint_to_image_coords(
             keypoint: List[float], image_size: Tuple[int, int]
         ) -> Tuple[int, ...]:
             """Second helper function to convert relative keypoint coordinates to
             absolute image coordinates.
             Keypoint coords ranges from 0 to 1
             where (0, 0) = image top-left, (1, 1) = image bottom-right.

             Args:
                 bbox (List[float]): List of 2 floats x, y (relative)
                 image_size (Tuple[int, int]): Width, Height of image

             Returns:
                 Tuple[int, ...]: x, y in integer image coords
             """
             width, height = image_size[0], image_size[1]
             x, y = keypoint
             x *= width
             y *= height
             return int(x), int(y)

         def draw_text(img, x, y, text_str: str, color_code):
             """Helper function to call opencv's drawing function,
             to improve code readability in node's run() method.
             """
             cv2.putText(
                 img=img,
                 text=text_str,
                 org=(x, y),
                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.4,
                 color=color_code,
                 thickness=2,
             )

         class Node(AbstractNode):
             """Custom node to display keypoints and count number of hand waves

             Args:
                 config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
             """

             def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
                 super().__init__(config, node_path=__name__, **kwargs)
                 # setup object working variables
                 self.right_wrist = None
                 self.direction = None
                 self.num_direction_changes = 0
                 self.num_waves = 0

             def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
                 """This node draws keypoints and count hand waves.

                 Args:
                     inputs (dict): Dictionary with keys
                         "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

                 Returns:
                     outputs (dict): Empty dictionary.
                 """
                 # get required inputs from pipeline
                 img = inputs["img"]
                 bboxes = inputs["bboxes"]
                 bbox_scores = inputs["bbox_scores"]
                 keypoints = inputs["keypoints"]
                 keypoint_scores = inputs["keypoint_scores"]

                 img_size = (img.shape[1], img.shape[0])  # image width, height

                 # get bounding box confidence score and draw it at the
                 # left-bottom (x1, y2) corner of the bounding box (offset by 30 pixels)
                 the_bbox = bboxes[0]  # image only has one person
                 the_bbox_score = bbox_scores[0]  # only one set of scores

                 x1, y1, x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
                 score_str = f"BBox {the_bbox_score:0.2f}"
                 cv2.putText(
                     img=img,
                     text=score_str,
                     org=(x1, y2 - 30),  # offset by 30 pixels
                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale=1.0,
                     color=WHITE,
                     thickness=3,
                 )

                 # hand wave detection using a simple heuristic of tracking the
                 # right wrist movement
                 the_keypoints = keypoints[0]  # image only has one person
                 the_keypoint_scores = keypoint_scores[0]  # only one set of scores
                 right_wrist = None
                 right_shoulder = None

                 for i, keypoints in enumerate(the_keypoints):
                     keypoint_score = the_keypoint_scores[i]

                     if keypoint_score >= THRESHOLD:
                         x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
                         x_y_str = f"({x}, {y})"

                         if i == KP_RIGHT_SHOULDER:
                             right_shoulder = keypoints
                             the_color = YELLOW
                         elif i == KP_RIGHT_WRIST:
                             right_wrist = keypoints
                             the_color = YELLOW
                         else:  # generic keypoint
                             the_color = WHITE

                         draw_text(img, x, y, x_y_str, the_color)

                 if right_wrist is not None and right_shoulder is not None:
                     # only count number of hand waves after we have gotten the
                     # skeletal poses for the right wrist and right shoulder
                     if self.right_wrist is None:
                         self.right_wrist = right_wrist  # first wrist data point
                     else:
                         # wait for wrist to be above shoulder to count hand wave
                         if right_wrist[1] > right_shoulder[1]:
                             pass
                         else:
                             if right_wrist[0] < self.right_wrist[0]:
                                 direction = "left"
                             else:
                                 direction = "right"

                             if self.direction is None:
                                 self.direction = direction  # first direction data point
                             else:
                                 # check if hand changes direction
                                 if direction != self.direction:
                                     self.num_direction_changes += 1
                                 # every two hand direction changes == one wave
                                 if self.num_direction_changes >= 2:
                                     self.num_waves += 1
                                     self.num_direction_changes = 0  # reset direction count

                             self.right_wrist = right_wrist  # save last position
                             self.direction = direction

                     wave_str = f"#waves = {self.num_waves}"
                     draw_text(img, 20, 30, wave_str, YELLOW)

                 return {
                     "hand_direction": self.direction if self.direction is not None else "None",
                     "num_waves": self.num_waves,
                 }

   Note that we still use a custom node for the "wave" logic as we need to produce
   custom outputs such as ``hand_direction`` and ``num_waves``.

#. ``callbacks/sql_callback.py``:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for sql_callback.py**

      .. code-block:: python
         :linenos:

         """
         Callbacks for interacting with an external database.
         """

         import logging
         import sqlite3
         from datetime import datetime
         from typing import Any, Dict

         DB_FILE = "wave.db"  # name of database file


         class SQLCallback:
             """Interacts with a SQLite database and provides a callback method to
             trigger updates.
             """

             def __init__(self) -> None:
                 self.logger = logging.getLogger(__name__)
                 self.conn = None
                 try:
                     self.conn = sqlite3.connect(DB_FILE)
                     self.logger.info(f"Connected to {DB_FILE}")
                     sql = """ CREATE TABLE IF NOT EXISTS wavetable (
                                 datetime text,
                                 hand_direction text,
                                 wave_count integer
                            ); """
                     cur = self.conn.cursor()
                     cur.execute(sql)
                 except sqlite3.Error as e:
                     self.logger.info(f"SQL Error: {e}")

             def track_wave_callback(self, data_pool: Dict[str, Any]) -> None:
                 hand_direction = data_pool["hand_direction"]
                 num_waves = data_pool["num_waves"]
                 self.update_db(hand_direction, num_waves)

             def update_db(self, hand_direction: str, num_waves: int) -> None:
                 """Helper function to save current time stamp, hand direction and
                 wave count into DB wavetable.
                 """
                 now = datetime.now()
                 dt_str = f"{now:%Y-%m-%d %H:%M:%S}"
                 sql = """ INSERT INTO wavetable(datetime,hand_direction,wave_count)
                           values (?,?,?) """
                 cur = self.conn.cursor()
                 cur.execute(sql, (dt_str, hand_direction, num_waves))
                 self.conn.commit()


         sql_callback_obj = SQLCallback()

   We have recreated the ``output.sqlite`` node into a standalone class. As it no
   longer inherits from ``AbstractNode``, having a ``run()`` method is not necessary
   anymore. Instead, we implement a ``track_wave_callback()`` method which will
   be used update the database with the require information. As ``track_wave_callback()``
   is an instance method, we need to instantiate ``SQLCallback`` as ``sql_callback_obj``
   in order to use it.

#. ``pipeline_config.yml``:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.visual:
          source: wave.mp4
      - model.yolo
      - model.posenet
      - custom_nodes.dabble.wave:
          callbacks:
            run_end: ["sql_callback::sql_callback_obj::track_wave_callback"]
      - draw.poses
      - output.screen

   To use the callback function, we override the ``callbacks`` config option in
   ``custom_nodes.dabble.wave``. The key ``run_end`` specifies that the callback
   function will be invoked after the node's ``run()`` method. To specify the callback
   function to be invoked we format the string as ``<script name>::<instance name>::<instance method name>``.

Run this project with :greenbox:`peekingduck run`, and when completed, a ``wave.db``
database file will be generated. We can examine the created database using similar
commands as the :ref:`Interfacing with SQL<tutorial_sql>` tutorial:

   .. admonition:: Terminal Session

      | \ :blue:`[~user/wave_project]` \ > \ :green:`sqlite3` \
      | SQLite version 3.39.2 2022-07-21 15:24:47
      | Enter ".help" for usage hints.
      | Connected to a transient in-memory database.
      | Use ".open FILENAME" to reopen on a persistent database.
      | @sqlite> \ :green:`.open wave.db` \ 
      | @sqlite> \ :green:`.schema wavetable` \ 
      | CREATE TABLE wavetable (
      |                         datetime text,
      |                         hand_direction text,
      |                         wave_count integer
      |                    );
      | @sqlite> \ :green:`select * from wavetable where wave_count > 0 limit 5;` \ 
      | 2022-11-08 13:30:06|left|1
      | 2022-11-08 13:30:06|right|1
      | 2022-11-08 13:30:06|left|2
      | 2022-11-08 13:30:06|left|2
      | 2022-11-08 13:30:06|left|2
      | @sqlite> \ :green:`select * from wavetable order by datetime desc limit 5;` \ 
      | 2022-11-08 13:30:36|right|70
      | 2022-11-08 13:30:36|left|71
      | 2022-11-08 13:30:35|left|69
      | 2022-11-08 13:30:35|left|69
      | 2022-11-08 13:30:35|left|69

Press :greenbox:`CTRL-D` to exit from ``sqlite3``.

