# Guide: Building Custom Nodes to run with PeekingDuck

In this tutorial, we'll be building a custom csv_logger which writes data into a file. It serves as a data collection step that stores useful metrics into a csv file. This csv file can be used for logging purposes and can serve as a base for data analytics.

A step by step instruction to build this custom node is provided below. 

## Install PeekingDuck (Prerequisite)

1. Ensure that you have peekingduck installed by running `peekingduck --help`
2. If it wasn't already installed, follow the [installation guide](../README.md) 

## Step 1: Start a PeekingDuck Project

Once PeekingDuck has been installed, use `peekingduck init` to initialise a new project with our template:
```bash
> mkdir <project_name>
> cd <project_name>
> peekingduck init
> mkdir results
```

### Folder Structure
Your project folder should look like this at the end of the tutorial.

```bash
.
├── src
│   └── custom_nodes
│       ├── <node_type>               # <node_type> = "output" in this tutorial
│       │    ├── utils
│       │    │   └── csv.py           
│       │    └── csv_writer.py        
│       └── configs
│            └── <node_type>          # <node_type> = "output" in this tutorial
│                └── csv_writer.yml   
├── run_config.yml
├── results                           
```

- `src/custom_nodes` is where custom nodes created for the project should be housed.
- `run_config.yml` is the basic yml file to select nodes in the pipeline. You will be using this to run your peekingduck pipeline.

## Step 2: Populate Run Config

The `run_config.yml` will be created. Input the following lines of code into the file. Newly minted nodes needs to be identified by the following structure `custom_nodes.<node-type>.<node>` .


```yaml
# run_config.yml
nodes:
- input.live # -- or `input.recorded` to use your videofiles
- model.yolo
- heuristic.bbox_count
- draw.bbox
- draw.bbox_count
- output.screen
- custom_nodes.output.csv_writer
```

PeekingDuck is designed for flexibility and coherence between in-built and custom nodes; you can design a pipeline which has both types of nodes.

As PeekingDuck runs the pipeline sequentially, it is important to check if the nodes preceding your custom nodes provides the correct inputs to your node.

## Step 3: Create Node Configs

Create a custom node config under `src/custom_nodes/configs/<node_type>/<node>.yml`. In this tutorial it will be `src/custom_nodes/configs/output/csv_writer.yml` to list configurations required by the system.

```yaml
input: ["count"]                         # inputs required by the node
output: ["end"]                          # outputs of the node
period: 1                                # time interval (s) between logs
filepath: 'results/stats.csv'            # output file location
```

### Explanation

Node configs contains information on the input and outputs for PeekingDuck to manage.
We recommend new users to use the [config template](../peekingduck/configs/node_template.yml) for reference.

Your node config yaml file should contain the following:
- `input` (list of str): the key(s) to the required inputs for your node
- `output` (list of str): the key(s) to the outputs
- (optional) node-specific parameters. In our case, it will be `period`.

Note: While the keys for input and output can be arbitrary strings, keys should be consistent across all nodes in the pipeline.


## Step 4: Create your node scripts

We recommend new users to use the node template below.

1. Initialise your node script, `src/custom_nodes/<node_type>/<node>.py`, in our example it will be `src/custom_nodes/output/csv_writer.py`. You may want to use the following template start building your custom node.

```python
# this is a template for writing custom nodes
from peekingduck.pipeline.nodes.node import AbstractNode

class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path =__name__)

    def run(self, inputs):
        pass
```

2. Develop `run()`, the core function that PeekingDuck will call in the pipeline. Nodes can simply retrieve the necessary data by querying the input in a dict-like fashion. In this case, we take the input `count` and write the results into the specified filepath.

```python
# csv_writer.py
from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.csv import CSVLogger

class Node(AbstractNode):
    """Node that draws object counting"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        file_path = config["filepath"]
        inputs = config["input"].copy()
        period = config["period"]
        self.csv_logger = CSVLogger(file_path, inputs, period)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Method that draws the count on the top left corner of image

         Args:
             inputs (dict): Dict with keys "count" and "img".
         Returns:
             outputs (dict): None
         """

        self.csv_logger.write(inputs)
            
        return {}

    def __del__(self):
        del self.csv_logger
```

Note:
- Class name should be maintained as `Node`.
- PeekingDuck pipelines will feed a dict as an input to the node. In order to access the input data, simply call `inputs["in1"]`
- `run` must return a dictionary with the key-value pair defined as `{"out1": results_object}`. `out1` must be consistent with the outputs defined in the node configs. In this example, there is no output but we need to return an empty dictionary `{}`
- Logging is added to all nodes. To access it, simply call `self.logger` (e.g. `self.logger.info("Model loaded!")`)

## Step 5: Create Utilities

We recommend placing the utility files together with your node folder `src/custom_nodes/<node_type>/utils/<your-util>.py`. For this tutorial we will place the following code under `src/custom_nodes/output/utils/csv.py`. 

The implementation below uses `period` which dictates the time interval (in seconds) between each log entry.

```python
# csv.py
import csv
from datetime import datetime

class CSVLogger:
    """A class to log the chosen information into csv heuristic results"""

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
```

## Step 5: Run PeekingDuck!

### Final Checks

- By default, PeekingDuck assumes that your custom nodes are found in `src/custom_nodes`.

- Ensure that the files are in the correct [folder structure](#folder-structure).

- NOTE: Create a `results` folder at the same level as `src`!

### Running the Module

- Now the setup is complete and ready to run! Run the setup using the following code.

    `peekingduck run --config_path=run_config.yml`

- If the setup is working, you should see an output screen being displayed. Terminate the program by clicking on the output screen and pressing `q`.

- After the program finishes running, open the file at `results/stats.csv` to view the stored information.

**NOTE:**

While running, the csv file may be empty. This is because the implementation of this csv logger completes the writing at the end of the instruction.
