# Building Custom Nodes

## Creating nodes

All nodes should inherit `AbstractNode`. We recommend new users to use the [node template](../peekingduck/pipeline/nodes/node_template.py)

Things to note:
- logging is added to all nodes. To access it, simply call `self.logger` (e.g. `self.logger.info("Model loaded!")`)
- `run` is the core function for all nodes to run with PeekingDuck's pipeline.
  - pipeline will feed a dict as an input to the node. In order to access the input data, simply call `inputs["in1"]`
  - it must return a dictionary with the key-value pair defined as `{"out1": results_object}`
  - keys used for inputs and outputs must be consistent with the strings defined in the configs (See [configs](#configs))
- Class name should be maintained as `Node`.
- Your node should only be initialised with a `config: Dict[str, Any]` parameter. Any other parameters should be included into the config file.

## Node Configs
Node configs contains information on the input and outputs for PeekingDuck to manage. You may also add in node-specific parameters to control node behavior.

Nodes are able to access the parameters as a dictionary. For example, in [`input.live` node](../peekingduck/configs/input/live.yml), the node can access the resolution parameter by calling its key from configs:

```
self.resolution = config['resolution']
```

PeekingDuck only supports yaml at this point of time. New users may refer to [config template](../peekingduck/configs/node_template.yml) for reference.

## Organising your custom nodes in your repository

By default, PeekingDuck assumes that your custom nodes are found in `src/custom_nodes` in the following structure:

```
<project_name>

├── run_config.yml
└── src
    └── custom_nodes
        ├── node1.py
        ├── node1.yml
        ├── node2.py
        └── node2.yml
```

Note that the filenames for the node script, node configs, and node name in runner configs should be consistent. e.g. (The files associated with the Node `random_number_generator` should be `random_number_generator.py`, `random_number_generator.yml`, and `custom.random_number_generator` respectively.)

# Adding your custom nodes to the pipeline

## Classic mode (default; recommended)


## Notebook mode
TBD



## Contributions, Feedback

TBD