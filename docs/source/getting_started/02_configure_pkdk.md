# Changing Nodes and Settings

This page will guide users on how to control and configure how PeekingDuck behaves by
- Selecting which nodes to include in the pipeline
- Configuring node behaviour

You can refer to our [API Documentation](/peekingduck.pipeline.nodes) section to see the available nodes in PeekingDuck for selection and their respective configurable settings. Alternatively, to get a quick overview of Peekingduck's nodes, run the following command: 
 ```
 > peekingduck nodes
 ```

In this guide, we will make changes to PeekingDuck config files to run pose estimation models. We will also teach users how to make changes to the default configurations to run a bird detection pipeline on a local video file.

By default, `run_config.yml` uses the following nodes:

```yaml
nodes:
- input.live
- model.yolo
- draw.bbox
- output.screen
```

## Changing Nodes

1. Let's modify the default `run_config.yml` to run a pose estimation demo using the following nodes:
    ```yaml
    nodes:
      - input.live
      - model.posenet         # Changed to a pose estimation model
      - draw.poses            # Changed to draw pose skeletons instead of bounding boxes
      - output.screen
    ```

2. Now run PeekingDuck with `peekingduck run`. If you have a webcam, you should see the demo running live:

    <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/posenet_demo.gif" width="50%">

    Terminate the program by clicking on the output screen and pressing `q`. That's it! By changing two lines in `run_config.yml`, you now have a pose estimation pipeline running on your webcam.

## Configuring node behaviour

PeekingDuck is also able to work on recorded videos or saved images, and we'll use the `input.recorded` and `output.media_writer` nodes for that. For this demo, you'll have to [download](https://storage.googleapis.com/peekingduck/videos/ducks.mp4.zip) and unzip a short video of ducks, and use `model.yolo` again to detect them.

By default, `model.yolo` detects humans. We can change its behavior either by 1) updating the `run_config.yml`; or 2) updating the configs at runtime via CLI.


### via `run_config.yml`


1. From the default `run_config.yml`, we'll need to change the settings of 3 nodes: the `input` and `output` directories, and the object to be detected from a human to a bird, as follows:
    ```yaml
    nodes:
      - input.recorded:       # note the ":"
          input_dir: <directory where videos/images are stored>
      - model.yolo:           # note the ":"
          detect_ids: [14]    # ID to detect the "bird" class is 14 for this model
      - draw.bbox
      - output.media_writer:  # note the ":"
          output_dir: <directory to save results>
    ```

2. Run `peekingduck run`

### via CLI

1. From the default `run_config.yml`, update the nodes accordingly:
  ```yaml
    nodes:
    - input.recorded
    - model.yolo
    - draw.bbox
    - output.screen
    - output.media_writer
  ```
2. Run PeekingDuck with `--node_config` and the new configurations in a JSON-like structure:
 ```bash
 peekingduck run --node_config "{'input.recorded': {'input_dir': '<directory where videos/images are stored>'}, 'model.yolo': {'detect_ids': [14]}, 'output.media_writer': {'output_dir': '<directory to save results>'}}"
 ```

  Note the following:
  1. configs are structured in a {<node_name>: {<param_name>:<param_value>}} format.
  2. Unlike the yaml files, filepaths, and strings need to be encased with quotation marks. e.g. `'input_dir': '<directory/filepath>'` .
  3. For Windows users, use `\\` for directories/filepaths.
  4. PeekingDuck will only accept updates to nodes that are declared in `run_config.yml`. For nodes that are not in the node list, or for configs that are not relevant to the nodes, PeekingDuck will raise a warning and use defaults where applicable. 


Regardless of the method you choose to configure PeekingDuck, the processed files will be saved to the specified output directory once PeekingDuck is finished running. You should get this in your output file:

  <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/ducks_demo.gif" width="50%">



## PeekingDuck API Reference
We have highlighted the basic configurations for different nodes that you may wish to use for your project.
To find out what other settings can be tweaked for different nodes, check out the individual node configurations in PeekingDuck's [API Reference](/peekingduck.pipeline.nodes).
