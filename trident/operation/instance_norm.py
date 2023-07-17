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

import logging

import torch
import triton

from trident import kernel, language, util


class InstanceNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        (
            inp,
            run_mean,
            run_var,
            wgt,
            bis,
            use_input_stats,
            momentum,
            eps,
        ) = args

        if use_input_stats:
            InstanceNorm.__optimize(inp, run_mean, run_var, momentum)
            run_mean = run_var = None

        return InstanceNorm.__forward(
            inp,
            run_mean,
            run_var,
            wgt,
            bis,
            eps,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            inp,
            run_mean,
            run_var,
            wgt,
            bis,
            use_input_stats,
            momentum,
            eps,
        ) = inputs

        ctx.save_for_backward(inp, wgt, bis)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad_outputs):
        return InstanceNorm.__backward(*grad_outputs, *ctx.saved_tensors, ctx.eps)

    @staticmethod
    def __forward(inp, run_mean, run_var, wgt, bis, eps):
        assert inp.is_contiguous()

        num_bt, num_ch, vec_sz = inp.shape
        out = torch.empty_like(inp)

        def grid(meta):
            return [num_bt * num_ch]

        kernel.InstanceNorm.forward[grid](
            inp,
            num_ch,
            vec_sz,
            run_mean,
            run_var,
            wgt,
            bis,
            eps,
            out,
            util.block_size(vec_sz, inp.element_size()),
            util.map_dtype(inp.dtype),
            num_warps=util.num_warps(vec_sz, inp.element_size(), 4),
        )

        return out

    @staticmethod
    def __backward(grad_out, inp, wgt, bis, eps):
        num_bt, num_ch, vec_sz = inp.shape
        grad_inp = torch.empty_like(inp)
        grad_wgt = torch.zeros_like(wgt) if wgt is not None else None
        grad_bis = torch.zeros_like(bis) if bis is not None else None

        def grid(meta):
            return [num_bt * num_ch]

        kernel.InstanceNorm.backward[grid](
            grad_out,
            inp,
            grad_inp,
            num_ch,
            vec_sz,
            wgt,
            grad_wgt,
            grad_bis,
            eps,
            blk_sz=triton.next_power_of_2(vec_sz),
        )

        return grad_inp, None, None, grad_wgt, grad_bis, None, None, None, None

    @staticmethod
    def __optimize(inp, run_mean, run_var, momentum):
        if run_mean is None or run_var is None:
            return

        num_bt, num_ch, vec_sz = inp.shape
        mean = torch.zeros_like(run_mean)
        var = torch.zeros_like(run_var)

        def grid(meta):
            return [num_bt * num_ch]

        kernel.InstanceNorm.mean_var[grid](
            inp,
            num_bt,
            num_ch,
            vec_sz,
            mean,
            var,
            util.block_size(vec_sz, inp.element_size()),
            util.map_dtype(inp.dtype),
        )

        def grid(meta):
            return [1]

        kernel.InstanceNorm.optimize[grid](
            mean,
            run_mean,
            var,
            run_var,
            num_ch,
            momentum,
            triton.next_power_of_2(num_ch),
        )
