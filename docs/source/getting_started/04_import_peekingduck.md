# Import PeekingDuck as a Module

PeekingDuck is designed to be flexible enough for users to deploy PeekingDuck as an app,
extend it for custom use cases, or just use part of it for prototyping purposes.

Here are some examples where you might want to import PeekingDuck as a module:

1. Wrap a FastAPI app around a Object Detection module
2. Integrate PeekingDuck into your existing python codebase
3. Use YOLOv4 for your project in your Jupyter Notebook


## Sample Code
To understand the building blocks, refer to the Introduction [here.](../index.md#how-peekingduck-works)


You may also refer to our API Reference to understand PeekingDuck's capabilities [here](/peekingduck.pipeline.nodes).

Running it in your python scripts can be simple:

```python
from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import recorded
from peekingduck.pipeline.nodes.model import yolo
from peekingduck.pipeline.nodes.draw import bbox
from peekingduck.pipeline.nodes.output import media_writer

input_dir = None  # Change this as desired
ouput_dir = None  # Change this as desired

# Initialise the nodes
input_node = recorded.Node(input_dir=input_dir)
yolo_node = yolo.Node()
draw_node = bbox.Node()
output_node = media_writer.Node(output_dir=output_dir)

# Run it in the runner
runner = Runner(nodes=[input_node, yolo_node, draw_node, output_node])
runner.run()

#Inspect the data
runner.pipeline.data

```

## Demonstration

We have an in-depth demonstration for running PeekingDuck on Google Colab [here](https://colab.research.google.com/drive/1gC_qaBSZsyGM1T-L_Vzo_3il44sJBJ2M).