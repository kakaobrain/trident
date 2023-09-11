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


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, weight, bias, eps, running_mean, running_var = args

        util.push_trace("BatchNorm.__forward")
        output = BatchNorm.__forward(input, weight, bias, eps, running_mean, running_var)
        util.pop_trace()

        ctx.save_for_backward(input, weight, bias)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        util.push_trace("BatchNorm.__backward")
        grad_input, grad_weight, grad_bias = BatchNorm.__backward(*grad_outputs, *ctx.saved_tensors, ctx.eps)
        util.pop_trace()

        return grad_input, grad_weight, grad_bias, None, None, None

    @staticmethod
    def __forward(inp, wgt, bis, eps, running_mean, running_var):
        bt_sz, vec_sz = inp.shape

        def grid(meta):
            return (vec_sz,)

        out = torch.empty_like(inp)
        bt_blk_sz = triton.next_power_of_2(bt_sz)

        util.push_trace("kernel.BatchNorm.forward")
        kernel.BatchNorm.forward[grid](
            inp,
            vec_sz,
            bt_sz,
            wgt,
            bis,
            eps,
            running_mean,
            running_var,
            out,
            blk_sz=bt_blk_sz,
        )
        util.pop_trace()

        return out

    @staticmethod
    def __backward(grad_out, inp, wgt, bis, eps):
        bt_sz, vec_sz = inp.shape

        def grid(meta):
            return (vec_sz,)

        grad_inp = torch.empty_like(inp)
        grad_wgt = torch.empty_like(wgt) if wgt is not None else None
        grad_bis = torch.empty_like(bis) if bis is not None else None
        bt_blk_sz = triton.next_power_of_2(bt_sz)

        util.push_trace("kernel.BatchNorm.backward")
        kernel.BatchNorm.backward[grid](
            grad_out,
            inp,
            grad_inp,
            vec_sz,
            bt_sz,
            wgt,
            grad_wgt,
            grad_bis,
            eps,
            blk_sz=bt_blk_sz,
        )
        util.pop_trace()

        return grad_inp, grad_wgt, grad_bis
