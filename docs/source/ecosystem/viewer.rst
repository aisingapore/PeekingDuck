******************
PeekingDuck Viewer
******************

.. include:: /include/substitution.rst

.. _pkd_viewer:

The PeekingDuck Viewer offers you an interactive GUI application to manage and run
PeekingDuck pipelines, and to view and analyze the output video.


Running the Viewer
------------------

The PeekingDuck Viewer can be activated using the CLI ``--viewer`` option:

   .. admonition:: Terminal Session

      | \ :blue:`[~user]` \ > \ :green:`peekingduck run \-\-viewer` \

A screenshot of the Viewer and its GUI components is shown below:

   .. figure:: /assets/tutorials/viewer_main.png
      :alt: PeekingDuck Viewer Screenshot

      The PeekingDuck Viewer screen, with explanations for the main controls.

Once the Viewer screen appears, PeekingDuck will begin executing the current pipeline.  
The pipeline output is displayed as a video in the center of the screen, with a progress
bar below it.

If pipeline input length is deterministic (e.g. using a video file as the source), the
progress bar functions like a normal progress bar moving from start to end.  
Upon completion, the progress bar will be replaced with a slider that you can use 
to navigate the output video.

If the length is non-deterministic (e.g. capturing a webcam video), then the progress bar
will function in a non-deterministic manner: animating itself to indicate progress
but without an end point (as PeekingDuck has no idea how long the webcam video will be). 
In this case, click the ``Play/Stop`` button to end the webcam video capture, and the
progress bar will become a slider.


Navigating the Output Video
---------------------------

You can examine the output video of the executed pipeline by using the
``Play/Stop`` button to replay the entire video. 

You may also scrub through the video using the slider to go directly to the frames 
of interest.
The current video frame number is shown to the right of the slider, serving as a
position indicator.

The ``Zoom In`` and ``Zoom Out`` buttons allow you to adjust the video size. 


Using the Pipeline Playlist
---------------------------

Clicking the ``Playlist`` button will show/hide the playlist.

   .. figure:: /assets/tutorials/viewer_playlist.png
      :alt: PeekingDuck Viewer with Playlist Screenshot

      PeekingDuck Viewer with playlist shown.

The above screenshot shows the playlist on the right. 
The playlist is a collection of pipeline files that can be run with PeekingDuck. 
The current pipeline is automatically added to the playlist. 
This playlist is specific to you and is saved across different PeekingDuck Viewer runs.

Click to select a pipeline in the playlist. The pipeline's information will be displayed in 
the ``Pipeline Information`` panel below. It shows the pipeline's name, last modified date/time, 
and full file path.

To run the currently selected pipeline, click the ``Run`` button.

The ``Add`` button lets you manually add a pipeline file to the playlist. It will display 
a File Explorer dialog. Use it to select a PeekingDuck pipeline YAML file and it will be 
added to your playlist.

The ``Delete`` button will remove the currently selected pipeline from the playlist, after 
you have confirmed the deletion. 

A red ``X`` beside the pipeline indicates that the pipeline YAML file is missing. 
This could mean the pipeline had been added earlier, but its YAML file had since been 
deleted or moved to another folder. 
Delete the ``X`` pipeline entry to remove it from the playlist.

The list of pipelines can be sorted in reverse order by clicking the playlist header.

.. note::

   The playlist is saved in `~/.peekingduck/playlist.yml`, where `~` is the user's home folder.


Exiting the Viewer
------------------

To exit the Viewer, close the Viewer window.

