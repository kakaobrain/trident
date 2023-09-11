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

from typing import Any

import torch
import triton

from trident import kernel, util


class Argmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, dim = args

        util.push_trace("Argmax.__forward")
        output = Argmax.__forward(input, dim)
        util.pop_trace()

        return output

    @staticmethod
    def __forward(input: torch.Tensor, dim: torch.int32):
        factory_kwargs = {"device": input.device, "dtype": torch.int64}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.Argmax.forward")
        kernel.Argmax.forward[grid](
            output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            util.dtype(output.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return output
