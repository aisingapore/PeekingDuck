## Changing Nodes and Settings

`peekingduck init` command created the `run_config.yml` file, which is PeekingDuck's main configuration file and is responsible for:
- Selecting which nodes to include in the pipeline
- Configuring node behaviour

**1. Selecting which nodes to include in the pipeline**:

  - By default, `run_config.yml` uses the following nodes:
    ```
    nodes:
      - input.live
      - model.yolo
      - draw.bbox
      - output.screen
    ```
  - Now, let's modify it to run a pose estimation demo using the following nodes:
    ```
    nodes:
      - input.live
      - model.posenet
      - draw.poses
      - output.screen
    ```

  - Now run PeekingDuck with `peekingduck run`. If you have a webcam, you should see the demo running live:

    <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/posenet_demo.gif" width="50%">

    Terminate the program by clicking on the output screen and pressing `q`.

**2. Configuring node behaviour**:
- If you're not using a webcam, don't worry about missing out! PeekingDuck is also able to work on recorded videos or saved images, and we'll use the `input.recorded` and `output.media_writer` nodes for that. For this demo, you'll have to [download](https://peekingduck.blob.core.windows.net/videos/ducks.mp4.zip) and unzip a short video of ducks, and use `model.yolo` again to detect them.

- We'll need to change the settings of 3 nodes, in order to set the input and output directories, and also to alter the object to be detected from a human to a bird, as follows:
  ```
  nodes:
    - input.recorded:       # note the ":"
        input_dir: <directory where videos/images are stored>
    - model.yolo:           # note the ":"
        detect_ids: [14]    # ID to detect the "bird" class is 14 for this model
    - draw.bbox
    - output.media_writer:  # note the ":"
        output_dir: <directory to save results>
  ```
- Once PeekingDuck has finished running, the processed files will be saved to the specified output directory. Open the processed file and you should get this:

  <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/ducks_demo.gif" width="50%">

- To find out what other settings can be tweaked for different nodes, check out PeekingDuck's [API Reference](/peekingduck.pipeline.nodes).
