# Copyright 2023 ⓒ Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton

from trident import kernel, util


class Argmax(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, dim = args

        return Argmax.__forward(input, dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def __forward(input, dim):
        factory_kwargs = {"device": input.device, "dtype": torch.int64}
        y_size, x_size = input.shape

        if dim == 0:
            output_size = x_size
            size_along_dim = y_size
        else:
            output_size = y_size
            size_along_dim = x_size

        def grid(meta):
            return (output_size,)

        output = torch.empty(output_size, **factory_kwargs)

        kernel.Argmax.forward[grid](
            output,
            input,
            y_size,
            x_size,
            dim,
            util.block_size(size_along_dim, input.element_size()),
            util.dtype(input.dtype),
        )

        return output
