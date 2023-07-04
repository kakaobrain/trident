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

import functools

import torch
import triton

from trident import kernel


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return LayerNorm.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, norm_sh, wgt, bis, eps, mean, std = inputs
        ctx.save_for_backward(inp, wgt, bis, mean, std)
        ctx.norm_sh = norm_sh

    @staticmethod
    def backward(ctx, *grad_outputs):
        return LayerNorm.__backward(*grad_outputs, *ctx.saved_tensors, ctx.norm_sh)

    @staticmethod
    def __forward(inp, norm_sh, wgt, bis, eps, mean, std):
        vec_sz = LayerNorm.__get_vec_sz(norm_sh)
        num_vec = inp.numel() // vec_sz

        def grid(meta):
            return (num_vec,)

        out = torch.empty_like(inp)
        vec_bs = triton.next_power_of_2(vec_sz)

        kernel.LayerNorm.forward[grid](inp.view(num_vec, vec_sz), vec_sz, wgt, bis, eps, mean, std,
                                       out.view(num_vec, vec_sz), vec_bs=vec_bs)

        return out

    @staticmethod
    def __backward(grad_out, inp, wgt, bis, mean, std, norm_sh):
        vec_sz = LayerNorm.__get_vec_sz(norm_sh)
        num_vec = inp.numel() // vec_sz

        def grid(meta):
            return (num_vec,)

        grad_inp = torch.empty_like(inp)
        grad_wgt = None if wgt is None else torch.zeros_like(wgt)
        grad_bis = None if bis is None else torch.zeros_like(bis)
        vec_bs = triton.next_power_of_2(vec_sz)

        # TODO: Create a tensor in a kernel after a bug of Triton is fixed.
        wgt = torch.zeros(vec_sz, device='cuda').fill_(1) if wgt is None else wgt

        kernel.LayerNorm.backward[grid](grad_out, inp, grad_inp, vec_sz, wgt, grad_wgt, grad_bis, mean, std,
                                        vec_bs=vec_bs)

        return grad_inp, None, grad_wgt, grad_bis, None, None, None, None,

    @staticmethod
    def __get_vec_sz(sh):
        return functools.reduce(lambda x, y: x * y, sh)
