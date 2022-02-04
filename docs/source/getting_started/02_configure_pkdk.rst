***************************
Changing Nodes and Settings
***************************

This page will guide users on how to control and configure how PeekingDuck behaves by

* Selecting which nodes to include in the pipeline
* Configuring node behavior

You can refer to our :ref:`API Documentation <api_doc>` section to see the available nodes in
PeekingDuck for selection and their respective configurable settings. Alternatively, to get a quick
overview of Peekingduck's nodes, run the following command::
   
    > peekingduck nodes

In this guide, we will make changes to PeekingDuck config files to run pose estimation models. We
will also teach users how to make changes to the default configurations to run a bird detection
pipeline on a local video file.

By default, ``run_config.yml`` uses the following nodes:

.. code-block:: yaml

   nodes:
   - input.live
   - model.yolo
   - draw.bbox
   - output.screen

Changing Nodes
==============

#. Let's modify the default ``run_config.yml`` to run a pose estimation demo using the following nodes:

   .. code-block:: yaml

      nodes:
      - input.live
      - model.posenet # Changed to a pose estimation model
      - draw.poses    # Changed to draw pose skeletons instead of bounding boxes
      - output.screen

#. Now run PeekingDuck with ``peekingduck run``. If you have a webcam, you should see the demo
   running live:

   .. image:: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/posenet_demo.gif
      :width: 50 %

   Terminate the program by clicking on the output screen and pressing ``q``. That's it! By
   changing two lines in ``run_config.yml``, you now have a pose estimation pipeline running on
   your webcam.

Configuring node behavior
=========================

PeekingDuck can also work on recorded videos or saved images, and we'll use the
:mod:`input.recorded` and :mod:`output.media_writer` nodes for that. For this demo, you'll have to
download and unzip `a short video of ducks <https://storage.googleapis.com/peekingduck/videos/ducks.mp4.zip>`_,
and use :mod:`model.yolo` again to detect them.

By default, :mod:`model.yolo` detects humans. We can change its behavior by either updating the
``run_config.yml`` or updating the configs at runtime via command line.

Via ``run_config.yml``
----------------------

#. From the default ``run_config.yml``, we'll need to change the settings of 3 nodes: the ``input``
   and ``output`` directories, and the object to be detected from a human to a bird, as follows:
   
   .. code-block:: yaml
      
      nodes:
      - input.recorded:      # note the ":"
          input_dir: <directory where videos/images are stored>
      - model.yolo:          # note the ":"
          detect_ids: [14]   # ID to detect the "bird" class is 14 for this model
      - draw.bbox
      - output.media_writer: # note the ":"
          output_dir: <directory to save results>

#. Run PeekingDuck with ``peekingduck run``

Via Command Line
----------------

#. From the default ``run_config.yml``, update the nodes accordingly:

   .. code-block:: yaml
      
      nodes:
      - input.recorded
      - model.yolo
      - draw.bbox
      - output.screen
      - output.media_writer

#. Run PeekingDuck with a ``--node_config`` flag and the new configurations in a JSON-like structure::
   
    peekingduck run --node_config "{'input.recorded': {'input_dir': '<directory where videos/images are stored>'}, 'model.yolo': {'detect_ids': [14]}, 'output.media_writer': {'output_dir': '<directory to save results>'}}"

   .. note::

      #. Configuration updates via command line are structured in a
         ``{<node_name>: {<param_name>:<param_value>}}`` format.
      #. Unlike in YAML files, filepaths and strings need to be encased with quotation marks, e.g.,
         ``'input_dir': '<directory/filepath>'``.
      #. For Windows users, use ``\\`` as the path separator for directories/filepaths.
      #. PeekingDuck will only accept updates to nodes that are declared in ``run_config.yml``. For
         nodes that are not declared, or for configs that are not relevant to the nodes,
         PeekingDuck will display a warning and use default settings where applicable.

Regardless of the method you used to configure PeekingDuck, the processed files will be saved to
the specified output directory once PeekingDuck has finished running. You should get this in your
output file:

.. image:: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/ducks_demo.gif
   :width: 50 %

PeekingDuck API Reference
=========================

We have highlighted the basic configurations for various nodes that you may wish to use for your
project. To find out what other settings can be tweaked for different nodes, check out the
individual node configurations in PeekingDuck's :ref:`API Documentation <api_doc>`.
