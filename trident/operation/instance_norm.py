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

from trident import kernel


class InstanceNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        x, eps = args

        assert x.dim() == 3 and x.is_cuda and x.is_contiguous()

        num_batches, num_channels, num_elements = x.shape
        y = torch.empty_like(x)

        assert y.is_cuda and x.is_contiguous

        grid = lambda meta: (num_batches, num_channels,)
        kernel.instance_norm_forward[grid](x, x.stride(0), x.stride(1),
                                           y, y.stride(0), y.stride(1),
                                           num_elements, eps,
                                           block_size=triton.next_power_of_2(num_elements))

        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("The backward of Instance Norm isn't implemented.")
