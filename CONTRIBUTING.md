# How to Contribute Nodes

Thank you for considering contributing to PeekingDuck! By contributing nodes, you'll be part of a community helping to expand the capabilities of PeekingDuck over time. The following sections describe the contribution process and guidelines.


## 1. Process

- **`dev` is our main development branch.** <br>
Any new nodes should be created on a new feature branch based on `dev`. Once completed, create a Pull Request for it, and a maintainer will review your code.


## 2. Guidelines

We use:
- Python PEP 8 convention
- Pylint to lint our code
- [PEP 484](https://www.python.org/dev/peps/pep-0484/) type hinting for functions and methods
- Each code file should have a copyright and license header (if using VS Code, you could use this [extension](https://marketplace.visualstudio.com/items?itemName=minherz.copyright-inserter) instead of copy and pasting the text below):

    ```
    Copyright 2021 AI Singapore

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ```

## 3. Code Styles

- **Comments**<br>
Use comments sparingly, it should be used for situations where the code is not self-explanatory. If and when comments are used, it should be clear and succinct.
- **Single Responsibility Principle**<br>
The function/method should only perform one role.
- **Don't Repeat Yourself (DRY)**<br>
When a group of code is being used multiple times. It is recommended to convert this group of code into functions or methods. For this project, keep the repeat in code to fewer than 3 times. If there exists codeblocks being reused multiple times, it is time to clean the code.
- **Naming**<br>
    - Use snake_case for python variables, methods, functions and filenames.
    - Use PascalCase for python classes
    - camelCase should be used only for Javascript/ 
    - Name variables explicitly so that it is easier to follow the code
        ```
        # bad
        for item in item_list:
            apply_model(item)...

        # better
        for frame in frames:
            apply_model(frame)
        ```
    - Use nouns for variables, e.g. `reader`
    - Use verbs for methods,e.g. `read`


## 4. How to Add a Node

1. First, what kind of node do you want to add? There are 6 categories of nodes `{node_type}` in PeekingDuck:
    - input
        - Takes in live video feeds or recorded videos/images
    - model
        - AI models such as YOLO
    - heuristic
        - Approximations such as estimating 3D distance from 2D images
    - usecase
        - Real life use cases such as people counting
    - draw
        - Superimposes the outputs of the model, heuristic, usecase to the image. Examples include bounding boxes, text, etc.
    - output
        - Useful output such as showing the result to screen, saving to database, posting to API, etc

Let's use a hypothetical example, where you'd like to add an AI model called `quack`. `quack` is an object detector that was specifically trained to detect ducks, and returns coordinates of bounding boxes around ducks if present in images. Thus, the `{node_type}` for `quack` is `model`.

2. Next, think of a succinct name for the node, `{node_name}`. In this example, we could simply call it `quack`. Bear in mind that nodes are added to the pipeline to be run in the [run_config.yml](run_config.yml) configuration file with this convention: `{node_type}`.`{node_name}` (separated by a dot), thus it needs to be be easy to tell what the node does from its name.

    ```
    # Examples of {node_type}.{node_name} convention:
    input.live
    model.quack
    draw.bbox
    output.screen
    ```


3. In PeekingDuck's pipeline, the output of one node would be the input to another. For example, the input to `model.quack` could be an image from `input.live`, and the output of `model.quack` would be bounding box coordinates that go to `draw.bbox`. Refer to our [Node Zoo](Nodezoo) for a full list of nodes that can be used in conjunction with yours. 

4. In [peekingduck/configs](peekingduck/configs), create a configuration file named `{node_type}`\_`{node_name}.yml` (separated by an underscore instead of a dot this time). In our case, the name would be `model_quack.yml`. The file should contain:
    ```
    # Mandatory
    input: ["img"]  # example where the input of this node is an image
    output: ["bboxes"] # example where the output of this node are bboxes

    # Optional depending on node
    threshold: 0.5 # example
    ```

5. Create the node in [peekingduck/pipeline/nodes/](peekingduck/pipeline/nodes){`node_type`}/{`node_name`}.py. In this example, it is 'peekingduck/pipeline/nodes/model/quack.py'. You can start with the following scaffold code:
    ```
    from peekingduck.pipeline.nodes.node import AbstractNode

    class Node(AbstractNode):
        def __init__(self, config):
            super().__init__(config, name=# replace with {node_type}.{node_name})

            # your code to init any necessary items

        def run(self, inputs: Dict):

            # your code to do something with inputs, and return outputs
            
            return outputs
    ```
    Things to note:
    - All nodes should inherit from the [AbstractNode class](peekingduck/pipeline/nodes/node.py)
    - Refer to the [Node Zoo](Nodezoo) for definitions of `self.inputs` and `self.outputs`. The types need to match the inputs and outputs fields in the '`{node_type}`\_`{node_name}.yml`' configuration file
    - Logging convention (to be updated)

6. If the quack node requires some utility functions, create a {`node_name`} folder and put the files with these utilities inside:
    ```
    model
    |- quack.py
    |- quack
    |    |- model.py
    |
    |- yolo.py
    |- yolo
    |    |- utils.py
    ```

7. After adding to the scaffold code, quack.py would look like this:
    ```
    from peekingduck.pipeline.nodes.node import AbstractNode
    from .quack.model import QuackModel


    class Node(AbstractNode):
        def __init__(self, config):
            super().__init__(config, name='model.quack')
            self.model = QuackModel(config)

        def run(self, inputs: Dict):

            results, _, _ = self.model.predict(inputs[self.inputs[0]])
            outputs = {self.outputs[0]: results}
            return outputs
    ```

8. Create a unit test for your node using pytest (to be updated).


