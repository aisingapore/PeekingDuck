# Guide: Building Custom Nodes to run with PeekingDuck

## Starting a PeekingDuck Project

Once PeekingDuck has been installed, use `peekingduck init` to initialise a new project with our template:
```bash
> mkdir <new_project>
> cd <new_project>
> peekingduck init
```

Your project folder should look like this:
```
<project_name>
├── run_config.yml
└── src
    └── custom_nodes
```

- `src/custom_nodes` is where custom nodes created for the project should be housed.
- `run_config.yml` is the basic yml file to select nodes in the pipeline. You will be using this to run your peekingduck pipeline.

## Creating nodes

We will guide you to create your first node `multiplier`, a node that takes in a number and multiplies it by `multiple`, a pre-defined parameter.

### Step 1: Create your node configs
Node configs contains information on the input and outputs for PeekingDuck to manage.
We recommend new users to use the [config template](../peekingduck/configs/node_template.yml) for reference.

Your node config yaml file should contain the following:
- `input` (list of str): the key(s) to the required inputs for your node
- `output` (list of str): the key(s) to the outputs
- (optional) node-specific parameters. In our case, it will be `multiple`.

Note: While the keys for input and output can be arbitrary strings, keys should be consistent across all nodes in the pipeline.

Here's what the configs for `multiplier`, `multiplier.yml` will look like:

```
input: ['number']
output: ['multiplied_number']

multiple: 2
```


### Step 2: Create your node scripts

We recommend new users to use the [node template](../peekingduck/pipeline/nodes/node_template.py)

1. Initialise your node script, `multiplier.py`, with the template:

```python
# multiplier.py
from peekingduck.pipeline.nodes.node import AbstractNode

class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path =__name__)


    def run(self, inputs):
        pass
```

2. Develop `run`, the core function that PeekingDuck will call in the pipeline. Nodes can simply retrieve the necessary data by querying the input in a dict-like fashion. In this case, we are taking the input `number`, multiplying it by `multiple`, and returning the results as `multiplied_number`:

```python
# multiplier.py
from peekingduck.pipeline.nodes.node import AbstractNode

class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path =__name__)

        self.multiple = config['multiple']

    def run(self, inputs):
        results = inputs['number'] * self.multiple
        output = {'multiplied_number': results}
        return output
```

Note:
- Class name should be maintained as `Node`.
- PeekingDuck pipelines will feed a dict as an input to the node. In order to access the input data, simply call `inputs["in1"]`
- `run` must return a dictionary with the key-value pair defined as `{"out1": results_object}`. `out1` must be consistent with the outputs defined in the node configs. In this case, it will be `{'multiplied_number': results}`
- logging is added to all nodes. To access it, simply call `self.logger` (e.g. `self.logger.info("Model loaded!")`)

### Step 3: Organise your custom nodes in your repository

By default, PeekingDuck assumes that your custom nodes are found in `src/custom_nodes` in the following structure:

```
<project_name>
├── run_config.yml
└── src
    └── custom_nodes
        ├── multiplier.py
        ├── multiplier.yml
        ├── node2.py
        └── node2.yml
```

Note that the filenames for the node script, node configs, and node name in runner configs should be consistent. e.g. The files associated with the Node `multiplier` should be `multiplier.py`, `multiplier.yml`, and `custom.multiplier` respectively.

### Step 4: Add your custom nodes to the runner configs
Add your custom node to `run_config.yml`, in the `custom.<node_name>` format.

As PeekingDuck runs the pipeline sequentially, it is important to check if the nodes preceding your custom nodes provides the correct inputs to your node.

In the case of `multiplier`, the `run_config.yml` may look like this:
```
nodes:
- random.random_number_generator         # outputs 'number'
- custom.multiplier
- output.file_writer
```


### Step 4b: Configure your nodes

If there are in-built nodes that you wish to configure, you may use `peekingduck get-configs` to compile a `node_config.yml` to your project repository.

Given a list of nodes (usually from `run_config.yml`), `get-configs` will retrieve and compile all associated configs into a single yaml. It will search the custom nodes folder for associated configs as well.


### Step 5: Run PeekingDuck!

### 5a. Classic mode (recommended)

You can run the pipeline with the following command:
```
peekingduck run
```


### 5b. Notebook mode
TBD



## Contributions, Feedback

TBD