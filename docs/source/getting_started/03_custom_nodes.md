# Building Custom Nodes to run with PeekingDuck

PeekingDuck is designed to work with custom use cases. This guide will showcase how anyone can develop custom nodes to be used with PeekingDuck.

In this tutorial, we'll be building a custom `csv_logger` which writes data into a file. It serves as a data collection step that stores useful metrics into a csv file. This csv file can be used for logging purposes and can serve as a base for data analytics. (PeekingDuck already has an [`output.csv_logger`](/peekingduck.pipeline.nodes.output.csv_writer.Node) node, but this is a good example for users to experience creating a custom node from scratch.)

A step by step instruction to build this custom node is provided below.

## Install PeekingDuck (Prerequisite)

1. Ensure that you have PeekingDuck installed by running `peekingduck --help`
2. If it wasn't already installed, follow the [installation guide](../getting_started/01_installation.md)

## Step 1: Start a PeekingDuck Project

Once PeekingDuck has been installed, initialise a new project with our template using `peekingduck init --custom_folder_name <custom_folder_name>`, where `<custom_folder_name>` is your desired folder name to house the custom nodes. If the argument is not provided, PeekingDuck will create the housing folder using the default name: `custom_nodes`.

```bash
> mkdir <project_name>
> cd <project_name>
> peekingduck init --custom_folder_name <custom_folder_name> # default is 'custom_nodes'
> mkdir results
```

Your project folder should look like this:

```bash
.
├── src
│   └── <custom_folder_name>
│       └── configs
├── run_config.yml
├── results
```

- `src/<custom_folder_name>` is where custom nodes created for the project should be housed.
- `run_config.yml` is the basic yaml file to select nodes in the pipeline. You will be using this to run your peekingduck pipeline.

## Step 2: Populate run_config.yml

The `run_config.yml` will be created by `peekingduck init`. For this tutorial, replace its contents with the following node list. Notice that newly minted nodes are identified by the following structure `<custom_folder_name>.<node-type>.<node>`. In this project, the `<custom_folder_name>` is named `custom_nodes`.

**IMPORTANT**: `<node_type>` needs to adhere to one of the five node types in PeekingDuck: `['input','output','model','dabble','draw']`


```yaml
# run_config.yml
nodes:
- input.live # -- or `input.recorded` to use your videofiles
- model.yolo
- dabble.bbox_count
- draw.bbox
- draw.legend
- output.screen
- custom_nodes.output.csv_writer
```

PeekingDuck is designed for flexibility and coherence between in-built and custom nodes; you can design a pipeline which has both types of nodes.

As PeekingDuck runs the pipeline sequentially, it is important to check if the nodes preceding your custom nodes provides the correct inputs to your node. PeekingDuck will return an error if the sequence of nodes are incorrect.

## Step 3: Create Node Configs

Create a custom node config under `src/<custom_folder_name>/configs/<node_type>/<node>.yml`. In this tutorial it will be `src/custom_nodes/configs/output/csv_writer.yml` to list configurations required by the system.

```yaml
input: ["count"]                # inputs required by the node
output: ["none"]                # outputs of the node.
period: 1                       # time interval (s) between logs
filepath: 'results/stats.csv'   # output file location
```

`results` directory was manually created in Step 1 and PeekingDuck will save the csv as `stats.csv` in the directory.
Node configs contains information on the input and outputs for PeekingDuck to manage.
We recommend new users to use the [config template](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/node_template.yml) for reference.

Your node config yaml file should contain the following:
- `input` (list of str): the key(s) to the required inputs for your node
- `output` (list of str): the key(s) to the outputs
- (optional) node-specific parameters. In our case, it will be `period` and `filepath`.

Note: While the keys for input and output can be arbitrary strings, keys should be consistent across all nodes in the pipeline.


## Step 4: Create your node scripts

We recommend new users to use the node template below.

1. Initialise your node script, `src/<custom_folder_name>/<node_type>/<node>.py`, in our example it will be `src/custom_nodes/output/csv_writer.py`. You may want to use the following template start building your custom node.

```python
# this is a template for writing custom nodes
from peekingduck.pipeline.nodes.node import AbstractNode

class Node(AbstractNode):
    def __init__(self, config = None, **kwargs):
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
             inputs (dict): Dict with keys "count".
         Returns:
             outputs (dict): None
         """

        self.csv_logger.write(inputs)

        return {}
```

Note:
- Class name should be maintained as `Node`.
- PeekingDuck pipelines will feed a dict as an input to the node. In order to access the input data, simply call `inputs["in1"]`
- `run` must return a dictionary with the key-value pair defined as `{"out1": results_object}`. `out1` must be consistent with the outputs defined in the node configs. In this example, there is no output but we need to return an empty dictionary `{}`
- Logging is added to all nodes. To access it, simply call `self.logger` (e.g. `self.logger.info("Model loaded!")`)

## Step 5: Create Utilities (Optional)

We recommend placing the utility files together with your node folder `src/<custom_folder_name>/<node_type>/utils/<your-util>.py`. For this tutorial we will place the following code under `src/custom_nodes/output/utils/csv.py`.

The implementation below uses `period` (declared in your configs) to `file_path` (also in configs) which dictates the time interval (in seconds) between each log entry.

```python
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
```

## Step 5: Run PeekingDuck!

### Final Checks

- By default, PeekingDuck assumes that your custom nodes are found in `src/<custom_folder_name>`.
- Create a `results` folder at the same level as `src`!
- Ensure that the files are in the correct folder structure.


Your repository should look similar to this:

```bash
.
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
```

### Running the Module

- Now the setup is complete and ready to run! Run the setup using the following code.

    `peekingduck run --config_path run_config.yml`

- If the setup is working, you should see an output screen being displayed. If you are using `input.live`, terminate the program by clicking on the output screen and pressing `q`.

- After the program finishes running, open the file at `results/stats.csv` to view the stored information.

**NOTE:**

While running, the csv file may be empty. This is because the implementation of this csv logger completes the writing at the end of the instruction.

