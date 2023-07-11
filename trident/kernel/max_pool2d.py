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

import triton

from trident import language


class MaxPool2d:
    @staticmethod
    @triton.jit
    def forward(
        inp_ptr,
        inp_ch,
        inp_w,
        inp_bt_st,
        inp_ch_st,
        inp_h_st,
        out_ptr,
        out_h,
        out_w,
        out_bt_st,
        out_ch_st,
        out_h_st,
        knl_sz,
        knl_bs: triton.language.constexpr,
        grp_sz: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        num_grp = language.cdiv(out_w, grp_sz)
        bt = language.batch(pid, inp_ch, out_h, num_grp)
        ch = language.channel(pid, inp_ch, out_h, num_grp)
        h = language.row(pid, out_h, num_grp)
        grp = language.col(pid, num_grp)
        w = grp * grp_sz

        inp_ptr += bt * inp_bt_st + ch * inp_ch_st + h * (knl_sz * inp_h_st) + w * knl_sz
        out_ptr += bt * out_bt_st + ch * out_ch_st + h * out_h_st + w

        inp_blk = language.make_conv2d_blk(1, inp_h_st, 1, knl_bs, knl_bs)
        inp_blk = triton.language.ravel(inp_blk)
        inp_blk = language.make_group_blk(inp_blk, grp_sz, knl_sz)
        inp_msk = language.make_conv2d_msk(1, knl_sz, knl_sz, 1, knl_bs, knl_bs)
        inp_msk = triton.language.ravel(inp_msk)
        inp_msk = language.make_group_msk(inp_msk, grp_sz, w, out_h)
        out_blk = triton.language.arange(0, grp_sz)
        out_msk = triton.language.arange(0, grp_sz) + w < out_w

        inp = triton.language.load(inp_ptr + inp_blk, inp_msk, -float("inf"))
        out = triton.language.max(inp, axis=0)

        triton.language.store(out_ptr + out_blk, out, out_msk)
