# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib
import os
import sys
from src.model.yolox.exp import Exp


def get_exp_by_file(exp_file) -> Exp:
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name) -> Exp:
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


def get_exp(exp_file=None, exp_name=None) -> Exp:
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
