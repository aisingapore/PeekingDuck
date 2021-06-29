# Copyright 2021 AI Singapore
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
# Code of this file is mostly forked from
# [@xuannianz](https://github.com/xuannianz))

"""
Inject Tensorflow Keras into model workloads
"""


from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.utils.keras_utils \
    import inject_tfkeras_modules, init_tfkeras_custom_objects
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files \
    import efficientnet as model

EfficientNetB0 = inject_tfkeras_modules(model.efficientnet_b0)
EfficientNetB1 = inject_tfkeras_modules(model.efficientnet_b1)
EfficientNetB2 = inject_tfkeras_modules(model.efficientnet_b2)
EfficientNetB3 = inject_tfkeras_modules(model.efficientnet_b3)
EfficientNetB4 = inject_tfkeras_modules(model.efficientnet_b4)
EfficientNetB5 = inject_tfkeras_modules(model.efficientnet_b5)
EfficientNetB6 = inject_tfkeras_modules(model.efficientnet_b6)
EfficientNetB7 = inject_tfkeras_modules(model.efficientnet_b7)

preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()
