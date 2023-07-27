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

from trident import kernel, util


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return GroupNorm.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, num_groups, wgt, bis, eps = inputs
        ctx.save_for_backward(inp, wgt, bis)
        ctx.num_groups = num_groups
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad_outputs):
        return GroupNorm.__backward(
            *grad_outputs, *ctx.saved_tensors, ctx.num_groups, ctx.eps
        )

    @staticmethod
    def __forward(inp, num_groups, wgt, bis, eps):
        bt_sz, vec_sz = inp.shape
        new_inp = inp.view(bt_sz * num_groups, vec_sz // num_groups)

        out = torch.empty_like(new_inp)

        def grid(meta):
            return [bt_sz * num_groups]

        kernel.LayerGroupNorm.forward[grid](
            new_inp,
            vec_sz // num_groups,
            wgt,
            bis,
            eps,
            out,
            num_groups,
            blk_sz=util.block_size(vec_sz // num_groups, new_inp.element_size()),
            dtype=util.dtype(new_inp.dtype),
            num_warps=util.num_warps(vec_sz // num_groups, new_inp.element_size()),
        )

        return out.view(bt_sz, vec_sz)

    @staticmethod
    def __backward(grad_out, inp, wgt, bis, num_groups, eps):
        bt_sz, vec_sz = inp.shape
        new_inp = inp.view(bt_sz * num_groups, vec_sz // num_groups)

        grad_inp = torch.empty_like(inp)
        grad_wgt = None if wgt is None else torch.zeros_like(wgt)
        grad_bis = None if bis is None else torch.zeros_like(bis)

        def grid(meta):
            return [bt_sz * num_groups]

        # TODO: Create a tensor in a kernel after a bug of Triton is fixed.
        wgt = torch.zeros(vec_sz, device="cuda").fill_(1) if wgt is None else wgt

        kernel.LayerGroupNorm.backward[grid](
            grad_out,
            new_inp,
            grad_inp,
            vec_sz // num_groups,
            wgt,
            grad_wgt,
            grad_bis,
            eps,
            num_groups,
            blk_sz=util.block_size(vec_sz // num_groups, inp.element_size()),
            dtype=util.dtype(inp.dtype),
            num_warps=util.num_warps(vec_sz // num_groups, inp.element_size()),
        )

        return (
            grad_inp,
            None,
            grad_wgt,
            grad_bis,
            None,
            None,
            None,
            None,
        )
