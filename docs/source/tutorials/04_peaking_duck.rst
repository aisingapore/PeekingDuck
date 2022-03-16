************
Peaking Duck
************

.. include:: /include/substitution.rst

PeekingDuck include some "power" nodes that are capable of processing the contents 
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
with a new custom :mod:`output` node that writes information into a local sqlite database.

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
      | Node directory:	/user/wave_project/src/custom_nodes
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
         from peekingduck.pipeline.nodes.node import AbstractNode
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
   data pool for subsequent consumption by the new ``output.sqlite`` custom node.


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

Type :greenbox:`CTRL-D` to exit from ``sqlite3``.





.. _tutorial_counting_cars:

Counting Cars
=============

This tutorial demonstrates using the :mod:`dabble.statistics` node to count the number
of cars travelling across a highway over time and the :mod:`draw.legend` node to display
the relevant statistics.

Create a new PeekingDuck project, download the `highway cars video
<http://orchard.dnsalias.com:8100/highway_cars.mp4>`_ and save it into the project
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
      - input.recorded:
          input_dir: highway_cars.mp4
      - model.yolo:
          detect_ids: ["car"]
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
Then, as these objects moved around in the video, they are identified based on 
their assigned identities and tracked according to their movements.

This tutorial demonstrates using :mod:`dabble.statistics` with a custom node to 
track the number of people walking down a path.

Create a new PeekingDuck project, download the `people walking video
<http://orchard.dnsalias.com:8100/people_walking.mp4>`_ and save it into the project
folder.

   .. admonition:: Terminal Session

      | \ :blue:`[~user]` \ > \ :green:`mkdir people_walking` \
      | \ :blue:`[~user]` \ > \ :green:`cd people_walking` \
      | \ :blue:`[~user/people_walking]` \ > \ :green:`peekingduck init` \

Create the following ``pipeline_config.yml``:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
          input_dir: people_walking.mp4
      - model.yolo:
          detect_ids: ["person"]
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

The above pipeline uses the Yolo model to detect people in the video and uses 
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
      | Node directory:	/user/people_walking/src/custom_nodes
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
      - input.recorded:
          input_dir: people_walking.mp4
      - model.yolo:
          detect_ids: ["person"]
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
This zone is also passed to our custom node ``dabble.filter_bbox`` for bounding box
filtering.
What ``dabble.filter_bbox`` will do is to take the list of bboxes as input and 
output a list of bboxes within the zone, dropping all bboxes outside it.
Then, :mod:`dabble.tracking` is used to track the people walking and 
:mod:`dabble.statistics` is used to determine the number of people walking in the zone,
by getting the maximum of the tracked IDs.
:mod:`draw.legend` has a new item :term:`zone_count` which displays the number of people 
walking in the zone currently.

The ``filter_bbox.yml`` and ``filter_bbox.py`` files are shown below:

**src/custom_nodes/configs/dabble/filter_bbox.yml**:

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

**src/custom_nodes/dabble/filter_bbox.py**:

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
         from peekingduck.pipeline.nodes.node import AbstractNode


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


