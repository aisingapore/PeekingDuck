**********
Power Duck
**********

.. include:: /include/substitution.rst

This section illustrates advanced power features such as object tracking
and the "power" nodes: ``dabble.statistics``, ``draw.legend``.


Object Tracking
===============

Object tracking is the application of CV models to automatically detect objects 
in a video and to assign a unique identity to each of them.
These objects can be either living (e.g. person) or non-living (e.g. car). 
Then, as these objects moved around in the video, they are identified based on 
their assigned identities and tracked according to their movements.

**[Todo] object tracking using JDE**


Power Nodes
===========

PeekingDuck include some power nodes that are capable of processing the contents 
or outputs of the other nodes and to accumulate information over time.
An example is the ``dabble.statistics`` node which can accumulate statistical 
information, such as calculating the mean and max of particular objects (like
people or cars).


**[Todo] ``dabble.statistics``**

**[Todo] ``draw.legend`` (?)**



Interfacing with SQL
====================

This tutorial demonstrates how to save data to an SQLite database.
We will extend the tutorial for :ref:`counting hand waves<count_hand_wave>` with
a new custom ``output`` node that writes information into a local sqlite
database.

First, create a new custom ``output.sqlite`` node in the ``custom_project``
folder:

.. admonition:: Terminal Session

   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck create-node` \
   | Creating new custom node...
   | Enter node directory relative to ~user/custom_project [src/custom_nodes]: \ :green:`⏎` \
   | Select node type (input, model, draw, dabble, output): \ :green:`output` \
   | Enter node name [my_custom_node]: \ :green:`sqlite` \
   |
   | Node directory:	/user/custom_project/src/custom_nodes
   | Node type:	output
   | Node name:	sqlite
   |
   | Creating the following files:
   |    Config file: ~user/custom_project/src/custom_nodes/configs/output/sqlite.yml
   |    Script file: ~user/custom_project/src/custom_nodes/output/sqlite.py
   | Proceed? [Y/n]: \ :green:`⏎` \
   | Created node!

The updated folder structure would be:

.. parsed-literal::

   \ :blue:`custom_project/` \ |Blank|
   ├── pipeline_config.yml
   ├── \ :blue:`src/` \ |Blank|
   │   └── \ :blue:`custom_nodes/` \ |Blank|
   │       ├── \ :blue:`configs/` \ |Blank|
   │       │   ├── \ :blue:`dabble/` \ |Blank|
   │       │   │   └── wave.yml
   │       │   └── \ :blue:`output/` \ |Blank|
   │       │       └── sqlite.yml
   │       ├── \ :blue:`dabble/` \ |Blank|
   │       │   └── wave.py
   │       └── \ :blue:`output/` \ |Blank|
   │           └── sqlite.py
   └── wave.mp4



Edit the following **5 files** as described below:


1. **src/custom_nodes/configs/output/sqlite.yml**:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["hand_direction", "num_waves"]
      output: ["none"]

      # No optional configs

The new ``output.sqlite`` custom node will take in the hand direction and the
current number of hand waves to save to the external database.


2. **src/custom_nodes/output/sqlite.py**:

   .. code-block:: python
      :linenos:

      """
      Custom node to save data to external database.
      """

      from typing import Any, Dict
      from datetime import datetime
      from peekingduck.pipeline.nodes.node import AbstractNode
      import sqlite3

      DB_FILE = "wave.db"


      class Node(AbstractNode):
         """Custom node to save hand direction and current wave count to database.

         Args:
            config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
         """

         def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
            super().__init__(config, node_path=__name__, **kwargs)

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

         def update_db(self, hand_direction: str, num_waves: int) -> None:
            now = datetime.now()
            dt_str = f"{now:%Y-%m-%d %H:%M:%S}"
            sql = """ INSERT INTO wavetable(datetime,hand_direction,wave_count) 
                      values (?,?,?) """
            cur = self.conn.cursor()
            cur.execute(sql, (dt_str, hand_direction, num_waves))
            self.conn.commit()

         def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            """This node does ___.

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

Line 10 specifies the name of the database file as ``wave.db``.

The node initialiser code in lines 23-35 will establish a connection to the
database and will create a table called ``wavetable`` if it does not exist.
This table will be used to store the hand direction and wave count data.
On first run, this code will also create the ``wave.db`` database file.

Lines 37-44 is a helper function ``update_db`` to update the database.
It saves the current date time stamp, hand direction and wave count into the 
``wavetable``.

Lines 56-58 of the node's ``run`` method retrieves the required inputs from the 
pipeline's data pool and calls ``self.update_db`` to save the data.


3. **src/custom_nodes/configs/dabble/wave.yml**:

   .. code-block:: yaml
      :linenos:

      # Dabble node has both input and output
      input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]
      output: ["hand_direction", "num_waves"]

      # No optional configs

To support the ``output.sqlite`` custom node's input requirements, we need to 
modify the ``dabble.wave`` custom node to return the current hand direction
``hand_direction`` and the current wave count ``num_waves``.


4. **src/custom_nodes/dabble/wave.py**:

   .. code-block:: python
      :lineno-start: 173

         ... same as previous ...
         return {
            "hand_direction": self.direction if self.direction is not None else "None",
            "num_waves": self.num_waves,
         }

This file is the same as the previous one, except for the changes to the last 
line 174 as shown above.
These changes outputs the ``hand_direction`` and ``num_waves`` to the pipeline's 
data pool for subsequent consumption.


5. **pipeline_config.yml**:

   .. code-block:: yaml
      :lineno-start: 10

      ... same as previous ...
      - custom_nodes.output.sqlite

The pipeline is the same as the previous one, except for the new line 11 that 
has been added to call the new custom node.

Run this project with ``peekingduck run`` and when completed, a new ``wave.db`` 
sqlite database file would be created in the current folder.
Examine the created database as follows:


.. admonition:: Terminal Session

   | \ :blue:`[~user/custom_project]` \ > \ :green:`sqlite3` \
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

Type ``CTRL-D`` to exit from ``sqlite3``.


.. note::

   The above tutorial assumes ``sqlite3`` has been installed in your system. |br|
   If your system does not have ``sqlite3``, please see the `SQLite Home Page 
   <http://www.sqlite.org/>`_ for installation instructions.









