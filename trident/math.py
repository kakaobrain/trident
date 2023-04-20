"""
Copyright 2023 ⓒ Kakao Brain Corp.

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

import torch
from trident import operation


class Softmax(torch.nn.Module):
    def __init__(self, dim=None):
        """
        Applies a softmax function to the input tensor. Output tensor in the range [0,1] and sum to 1.

        :param dim: A dimension along which softmax will be computed.
        """
        super().__init__()

        self.dim = dim

    def forward(self, x):
        return operation.Softmax.apply(x, self.dim)
