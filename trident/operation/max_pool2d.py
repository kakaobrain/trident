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

from trident import kernel, math


class MaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return MaxPool2d.__forward(*args)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def __forward(inp, knl_sz):
        assert inp.is_contiguous()

        inp_bt, inp_ch, inp_h, inp_w = inp.shape
        out_h = MaxPool2d.__get_out_size(inp_h, knl_sz)
        out_w = MaxPool2d.__get_out_size(inp_w, knl_sz)

        out = torch.empty(inp_bt, inp_ch, out_h, out_w, dtype=inp.dtype, device='cuda')

        assert out.is_contiguous()

        grid = lambda meta: (inp_bt * inp_ch * out_h * triton.cdiv(out_w, meta['grp_sz']),)
        grp_sz = math.clamp(128 // triton.next_power_of_2(knl_sz), 1, triton.next_power_of_2(out_w))

        kernel.MaxPool2d.forward[grid](inp, inp_ch, inp_w, inp.stride(0), inp.stride(1), inp.stride(2),
                                       out, out_h, out_w, out.stride(0), out.stride(1), out.stride(2),
                                       knl_sz, triton.next_power_of_2(knl_sz), grp_sz)

        return out

    @staticmethod
    def __get_out_size(num_elem, knl_sz):
        return ((num_elem - (knl_sz - 1) - 1) // knl_sz) + 1
