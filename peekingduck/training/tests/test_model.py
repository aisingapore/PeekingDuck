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

# test items:
# whether the pre-trained model can load the weights correctly by comparing the
# loaded weights with the ground truth weights. For PyTorch it is ordereddict.


# test the forward pass function
# def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
#     """Forward pass of the model."""
#     y = self.model(inputs)
#     print(f"X: {inputs.shape}, Y: {y.shape}")
#     print("Forward Pass Successful")
