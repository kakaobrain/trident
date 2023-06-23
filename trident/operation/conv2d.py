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


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return Conv2d.__forward(*args)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def __forward(inp, wgt, bis):
        assert inp.is_contiguous() and wgt.is_contiguous()

        inp_bt, inp_ch, inp_h, inp_w = inp.shape
        out_ch, wgt_ch, wgt_h, wgt_w = wgt.shape
        out_h = Conv2d.__get_out_size(inp_h, wgt_h, 1)
        out_w = Conv2d.__get_out_size(inp_w, wgt_w, 1)

        out = torch.zeros(inp_bt, out_ch, out_h, out_w, dtype=torch.float, device='cuda')

        assert out.is_contiguous()

        def grid(meta):
            return (inp_bt * out_ch * ((out_h + meta['grp_sz'] - 1) // meta['grp_sz']) * out_w,)

        kernel.Conv2d.forward[grid](inp, inp_ch, inp_h, inp_w, inp.stride(0), inp.stride(1), inp.stride(2),
                                    wgt, wgt_ch, wgt_h, wgt_w, wgt.stride(0), wgt.stride(1), bis,
                                    out, out_ch, out_h, out_w, out.stride(0), out.stride(1), out.stride(2),
                                    triton.next_power_of_2(wgt_ch), triton.next_power_of_2(wgt_h),
                                    triton.next_power_of_2(wgt_w), max(512 // triton.next_power_of_2(wgt_w), 1))

        return out

    @staticmethod
    def __get_out_size(in_size, wt_size, stride):
        return ((in_size - wt_size) // stride) + 1
