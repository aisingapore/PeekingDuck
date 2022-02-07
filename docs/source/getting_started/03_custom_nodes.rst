*********************************************
Building Custom Nodes to Run with PeekingDuck
*********************************************

You may need to create your own custom nodes. Perhaps you'd like to take a snapshot of a video
frame, and post it to your API endpoint; perhaps you have a model trained on a custom dataset, and
would like to use PeekingDuck's :ref:`input <api_doc>`, :ref:`draw <api_doc>`, and
:ref:`output <api_doc>` nodes. We've designed PeekingDuck to be very flexible --- you can create
your own nodes and use them with ours. This guide will showcase how anyone can develop custom nodes
to be used with PeekingDuck.

In this guide, we'll be building a custom ``csv_writer`` node which writes data into a file. It
serves as a data collection step that stores useful metrics into a CSV file. This CSV file can be
used for logging purposes and can serve as a base for data analytics. (PeekingDuck already has an
:mod:`output.csv_writer` node, but this is a good example for users to experience creating a custom
node from scratch.)

A step by step instruction to build this custom node is provided below.

Install PeekingDuck (Prerequisite)
==================================

#. Ensure that PeekingDuck is installed by running ``peekingduck --help``
#. If it isn't installed, follow the :doc:`installation guide </getting_started/01_installation>`

Step 1: Start a PeekingDuck Project
===================================

Once PeekingDuck has been installed, initialize a new project with our template using
``peekingduck init --custom_folder_name <custom_folder_name>``, where ``<custom_folder_name>`` is
your desired folder name to house the custom nodes. If the argument is not provided, PeekingDuck
will create the housing folder using the default name ``custom_nodes``.

.. code-block:: text

   > mkdir <project_name>
   > cd <project_name>
   > peekingduck init --custom_folder_name <custom_folder_name>
   > mkdir results

Your project folder should look like this::

    <project_name>
    ├── src
    │   └── <custom_folder_name>
    │       └── configs
    ├── run_config.yml
    ├── results

* ``src/<custom_folder_name>`` is where custom nodes created for the project should be housed.
* ``run_config.yml``` is the basic YAML file to select nodes in the pipeline. You will be using
  this to run your peekingduck pipeline.
* ``results`` is the directory to store outputs from the custom `csv_writer` node we will be
  creating.

Step 2: Populate ``run_config.yml``
===================================

The ``run_config.yml`` will be created by ``peekingduck init``. For this guide, replace its
contents with the following node list. Notice that newly minted nodes are identified by the
following structure ``<custom_folder_name>.<node-type>.<node>``. In this project, the
``<custom_folder_name>`` is named ``custom_nodes``.

.. note::

    ``<node_type>`` needs to adhere to one of the five node types in PeekingDuck:
    ``['input', 'output', 'model', 'dabble', 'draw']``

.. code-block:: yaml

   # run_config.yml
   nodes:
   - input.live # or `input.recorded` to use your video files
   - model.yolo
   - dabble.bbox_count
   - draw.bbox
   - draw.legend
   - output.screen
   - custom_nodes.output.csv_writer

PeekingDuck is designed for flexibility and coherence between built-in and custom nodes; you can
design a pipeline which has both types of nodes.

As PeekingDuck runs the pipeline sequentially, it is important to check if the nodes preceding your
custom nodes provides the correct inputs to your node. PeekingDuck will return an error if the
sequence of nodes are incorrect.

Step 3: Generate Template Node Config and Script
================================================

We recommend using the ``peekingduck create-node`` command with the ``--config_path`` flag to
generate the template config and script files for the custom nodes defined in your
``run_config.yml`` as demonstrated below::

    > peekingduck create-node --config_path run_config.yml

.. note::
   
   Nodes with badly formatted names and types will be skipped automatically when using
   ``peekingduck create-node`` with ``--config_path``.

.. seealso::

   You can use ``peekingduck create-node`` interactively and be prompted to fix any
   ill-formattings. Please see the :ref:`bonus section <create_node_interactive>` for more details.

If you have been following along to this guide, you should expect to see the following directory
structure::

    <project_name>
    ├── src
    │   └── custom_nodes
    │       ├── configs
    │       │   └── output
    │       │       └── csv_writer.yml
    │       └── output
    │           └── csv_writer.py
    ├── run_config.yml
    ├── results

Step 4: Create Node Config
==========================

Using the template config file generated from above, we can list configurations required by the
system in ``src/custom_nodes/configs/output/csv_writer.yml``.

.. code-block:: yaml

   input: ["count"] # inputs required by the node
   output: ["none"] # outputs of the node.
   period: 1 # time interval (s) between logs
   filepath: "results/stats.csv" # output file location

``results`` directory was manually created in Step 1 and PeekingDuck will save the CSV file as
``stats.csv`` in the directory. Node configs contain information on the input and output for
PeekingDuck to manage.

Your node config YAML file should contain the following:

* ``input`` (list of str): the key(s) to the required inputs for your node
* ``output`` (list of str): the key(s) to the outputs
* (optional) node-specific parameters. In our case, it will be ``period`` and ``filepath``.

.. note::
    
   The keys for input and output can be arbitrary strings, but they should be consistent across
   all nodes in the pipeline.

Step 5: Create Your Node Scripts
================================

#. You should see the following template code in the generated script file
   ``src/custom_nodes/output/csv_writer.py``.

   .. code-block:: python

      # csv_writer.py
      from typing import Any, Dict

      from peekingduck.pipeline.nodes.node import AbstractNode


      class Node(AbstractNode):
          """This is a template class of how to write a node for PeekingDuck.
          Args:
              config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
          """

          def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
              super().__init__(config, node_path=__name__, **kwargs)

              # initialize/load any configs and models here
              # configs can be called by self.<config_name> e.g. self.filepath
              # self.logger.info(f"model loaded with configs: config")

          def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
              """This node does ___.
              Args:
                  inputs (dict): Dictionary with keys "__", "__".
              Returns:
                  outputs (dict): Dictionary with keys "__".
              """

              # result = do_something(inputs["in1"], inputs["in2"])
              # outputs = {"out1": result}
              # return outputs

#. Develop ``run()``, the core function that PeekingDuck will call in the pipeline. Nodes can
   simply retrieve the necessary data by querying the input in a dict-like fashion. In this case,
   we take the input ``count`` and write the results into the specified filepath.

   .. code-block:: python

      # csv_writer.py
      from typing import Any, Dict

      from peekingduck.pipeline.nodes.node import AbstractNode

      from .utils.csv import CSVLogger

      class Node(AbstractNode):
          """Node that logs outputs of PeekingDuck and writes to a CSV"""

          def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
              super().__init__(config, node_path=__name__, **kwargs)
              file_path = self.filepath
              inputs = self.input.copy()
              period = self.period
              self.csv_logger = CSVLogger(file_path, inputs, period)

          def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
              """Method that draws the count on the top left corner of image

              Args:
                  inputs (dict): Dictionary with keys "count".
              Returns:
                  outputs (dict): None
              """

              self.csv_logger.write(inputs)

              return {}

   .. note::
       
      * Class name must be ``Node``.
      * PeekingDuck pipelines will feed a dict as an input to the node. In order to access the
        input data, simply call ``inputs["in1"]``
      * ``run()`` must return a dictionary with the key-value pair defined as
        ``{"out1": results_object}``. ``out1`` must be consistent with the outputs defined in the
        node configs. In this example, there is no output but we need to return an empty dictionary
        ``{}``
      * Logging is added to all nodes. To access it, simply call ``self.logger``, e.g.,
        ``self.logger.info("Model loaded!")``.

Step 6: Create Utilities (Optional)
===================================

We recommend placing the utility files together with your node folder
``src/<custom_folder_name>/<node_type>/utils/<your_util>.py``. For this guide we will place the
following code under ``src/custom_nodes/output/utils/csv.py``.

The implementation below uses ``period`` (declared in your configs) to ``file_path`` (also in
configs) which dictates the time interval (in seconds) between each log entry.

.. code-block:: python

   # csv.py
   import csv
   from datetime import datetime
   
   class CSVLogger:
       """A class to log the chosen information into csv dabble results"""
   
       def __init__(self, file_path, headers, period=1):
           headers.extend(["date"])
   
           self.csv_file = open(file_path, mode='a+')
           self.writer = csv.DictWriter(self.csv_file, fieldnames=headers)
           self.period = period
   
           # if file is empty write header
           if self.csv_file.tell() == 0: self.writer.writeheader()
   
           self.last_write = datetime.now()
   
       def write(self, content):
           curr_time = datetime.now()
           update_date = {"date": curr_time.strftime("%Y%m%d-%I:%M:%S")}
           content.update(update_date)
   
           if (curr_time - self.last_write).seconds >= int(self.period):
               self.writer.writerow(content)
               self.last_write = curr_time
   
       def __del__(self):
           self.csv_file.close()

Step 7: Run PeekingDuck!
========================

Final Checks
------------

* By default, PeekingDuck assumes that your custom nodes are found in ``src/<custom_folder_name>``.
* Create a ``results`` folder at the same level as ``src``!
* Ensure that the files are in the correct folder structure.

Your project folder should look similar to this::

    <project_name>
    ├── src
    │   └── <custom_folder_name>
    │       ├── output
    │       │    ├── utils
    │       │    │   └── csv.py
    │       │    └── csv_writer.py
    │       └── configs
    │            └── output
    │                └── csv_writer.yml
    ├── run_config.yml
    ├── results

Creating the Files Manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to set up the files manually, we recommend new users to use these
`config template <https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/node_template.yml>`_
and `script template <https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/pipeline/nodes/node_template.py>`_
for reference.

Running the Module
------------------

* Now the setup is complete and ready to run! Run the setup using the following code::

    peekingduck run --config_path run_config.yml
* If the setup is working, you should see an output screen being displayed. If you are using
  :mod:`input.live`, terminate the program by clicking on the output screen and pressing ``q``.
* After the program finishes running, open the file at ``results/stats.csv`` to view the stored
  information.

.. note::
    
    While running, the CSV file may be empty. This is because the implementation of this CSV logger
    completes the writing at the end of the instruction.

.. _create_node_interactive:

Bonus: Using ``peekingduck create-node`` Interactively
======================================================

``peekingduck create-node`` can be used interactively to create the template config and script
files for custom nodes as demonstrated below::

    > peekingduck create-node
    Creating new custom node...
    Enter node directory relative to /path/to/<project_name> [src/custom_nodes]: [Press Enter]
    Select node type (input, model, draw, dabble, output): output
    Enter node name [my_custom_node]: csv_writer

    Node directory:	/path/to/<project_name>/src/custom_nodes
    Node type:	output
    Node name:	csv_writer

    Creating the following files:
        Config file: /path/to/<project_name>/src/custom_nodes/configs/output/csv_writer.yml
        Script file: /path/to/<project_name>/src/custom_nodes/output/csv_writer.py
    Proceed? [Y/n]: [Press Enter]
    Created node!

Step a: Enter Your Custom Node Parent Directory
-----------------------------------------------

.. code-block:: text

   Enter node directory relative to /path/to/<project_name> [src/custom_nodes]: [Press Enter]

Enter the path of your custom node directory, ensure the path is relative to ``<project_name>``,
e.g., ``src/<custom_folder_name>``. The default value ``src/custom_nodes`` is used in this guide.

Step b: Select a Node Type for Your Custom Node
-----------------------------------------------

.. code-block:: text

    Select node type (input, model, draw, dabble, output): output

Select a node type from one of the five node types in PeekingDuck
``(input, model, draw, dabble, output)``. ``output`` type is selected in this guide.

Step c: Enter Your Custom Node Name
-----------------------------------

.. code-block:: text

    Enter node name [my_custom_node]: csv_writer

Enter a name for your custom node. Some checks are performed in the background to ensure that the
node name is valid and does not already exist (to prevent existing files from being overwritten).
The default value is ``my_custom_node`` but ``csv_writer`` is used in this guide.

Step d: Confirm Node Creation
-----------------------------

.. code-block:: text
        
    Node directory:	/path/to/<project_name>/src/custom_nodes
    Node type:	output
    Node name:	csv_writer

    Creating the following files:
        Config file: /path/to/<project_name>/src/custom_nodes/configs/output/csv_writer.yml
        Script file: /path/to/<project_name>/src/custom_nodes/output/csv_writer.py
    Proceed? [Y/n]: [Press Enter]

The full paths of the config and script files to be created will be shown for verification. You can
abort the process by entering ``n``. The default value of ``y`` is selected in this guide.

Alternative: Use `peekingduck create-node` with Command Line Options
--------------------------------------------------------------------

If you would like to speed things up a little and skip the interactive process, the command line
options ``--node_subdir``, ``--node_type``, and ``--node_name`` can be used with
``peekingduck create-node``. Steps a-c from above can be replicated with command line options as
demonstrated below::

    > peekingduck create-node --node_subdir src/custom_nodes --node_type output --node_name csv_writer
    Creating new custom node...

    Node directory:	/path/to/<project_name>/src/custom_nodes
    Node type:	output
    Node name:	csv_writer

    Creating the following files:
        Config file: /path/to/<project_name>/src/custom_nodes/configs/output/csv_writer.yml
        Script file: /path/to/<project_name>/src/custom_nodes/output/csv_writer.py
    Proceed? [Y/n]: [Press Enter]
    Created node!

A final confirmation is still required before the files are created. You can use any number and
combination of the available command-line options. You will be prompted for the missing values
through the same interactive process.
