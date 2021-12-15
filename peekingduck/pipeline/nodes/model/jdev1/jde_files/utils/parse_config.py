# Modifications copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2019 ZhongdaoWang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Interpret model config file.
"""

from typing import Any, List
from pathlib import Path


def parse_model_cfg(path: Path) -> List[Any]:
    """Parses the YOLOv3 layer configuration file and returns module definitions.

    Args:
        path (Path): Path to model definition config file.

    Returns:
        List[Any]: Model definitions from config file. Output is in the
            format of a list of dicts of model architecture.
    """
    with open(str(path), "r") as file:
        lines = file.read().split("\n")
        lines = [x for x in lines if x and not x.startswith("#")]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        module_defs: List[Any] = []
        for line in lines:
            if line.startswith("["):  # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]["type"] = line[1:-1].rstrip()
                if module_defs[-1]["type"] == "convolutional":
                    module_defs[-1]["batch_normalize"] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                if value[0] == "$":
                    value = module_defs[0].get(value.strip("$"), None)
                module_defs[-1][key.rstrip()] = value

        return module_defs
