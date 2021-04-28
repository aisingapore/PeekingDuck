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
- Absolute imports instead of relative imports:
    ```
    from peekingduck.pipeline.nodes.node import AbstractNode
    ```
- Each code file should have a copyright and Apache License header (if using VS Code, you could use this [extension](https://marketplace.visualstudio.com/items?itemName=minherz.copyright-inserter) instead of copy and pasting the text below):

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
    If you are re-using code from other open source repositories, do check that it has a **permissive license** such as Apache or MIT, and not a copyleft license such as GPL. The original license file, headers in python files (if any), notices (if any) should be copied over, and the following added underneath the original copyright statement:
    ```
    Modifications copyright 2021 AI Singapore
    
## 3. Code Styles

- **Comments**<br>
Use comments sparingly, it should be used for situations where the code is not self-explanatory. If and when comments are used, it should be clear and succinct.
- **Single Responsibility Principle**<br>
The function/method should only perform one role.
- **Naming**<br>
    - Use snake_case for python variables, methods, functions and filenames
    - Use PascalCase for python classes
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

1. First, what kind of node do you want to add? There are 5 categories of nodes `{node_type}` in PeekingDuck:
    - input
        - Takes in live video feeds or recorded videos/images
    - model
        - AI models such as YOLO
    - heuristic
        - Not an AI model, but does something useful from previous outputs, such as count the number of bounding boxes from YOLO
    - draw
        - Superimposes the outputs of the model and/or heuristic to the image. Examples include bounding boxes, text, etc.
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


3. In PeekingDuck's pipeline, the output of one node would be the input to another. For example, the input to `model.quack` could be an image from `input.live`, and the output of `model.quack` would be bounding box coordinates that go to `draw.bbox`. Refer to the [node data glossary](docs/node_data_glossary.md) for a full list of data keys that can be used in conjunction with yours. 

4. Create a configuration file for the node in [peekingduck/configs](peekingduck/configs)/{`node_type`}/{`node_name`}.yml. In our case, the name would be 'peekingduck/configs/model/quack.yml'. You can use this [template](peekingduck/configs/node_template.yml) as a guide. The file should contain:
    ```
    # Mandatory configs
    input: ["img"]  # example where the input of this node is an image
    output: ["bboxes"] # example where the output of this node are bboxes

    # Optional configs depending on node
    threshold: 0.5 # example
    ```

5. Create the node in [peekingduck/pipeline/nodes/](peekingduck/pipeline/nodes){`node_type`}/{`node_name`}.py. In this example, it is 'peekingduck/pipeline/nodes/model/quack.py'. You can start writing the node with this [template](peekingduck/pipeline/nodes/node_template.py).


    Things to note:
    - All nodes should inherit from the [AbstractNode class](peekingduck/pipeline/nodes/node.py)
    - Again, refer to the list of [node data glossary](docs/node_data_glossary.md) for possible inputs that can be used by your node
    - Use self.logger for any required logging, as shown in the [template](peekingduck/pipeline/nodes/node_template.py)

6. If the quack node requires some utility functions, create a folder (do not use {`node_name`} for the folder name) and put the files with these utilities inside:
    ```
    model
    |- quack.py
    |- quack_utils
    |    |- model.py
    |
    |- yolo.py
    |- yolov4
    |    |- utils.py
    ```

7. After editing the [template](peekingduck/pipeline/nodes/node_template.py), quack.py would look like this:
    ```
    """
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
    """

    from typing import Any, Dict
    from peekingduck.pipeline.nodes.node import AbstractNode

    from peekingduck.pipeline.nodes.model.quack_utils.model import QuackModel


    class Node(AbstractNode):
        """This node detects ducks in images."""
        def __init__(self, config: Dict[str, Any]) -> None:
            super().__init__(config, node_path=__name__)

            self.duck_thres = config["duck_thres"]
            self.model = QuackModel(self.duck_thres)
            self.logger.info("model loaded with configs: %s", self.duck_thres)


        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Finds bounding boxes around ducks, if any.

            Args:
                inputs (dict): Dict with keys "img".

            Returns:
                outputs (dict): Dict with keys "bboxes".
            """

            results, _, _ = self.model.predict(inputs["img"])
            outputs = {"bboxes": results}
            return outputs
    ```

8. Create a unit test for your node using pytest (to be updated).


