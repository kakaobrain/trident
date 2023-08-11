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
import triton.language as tl

from trident import language


class Conv2d:
    @staticmethod
    @triton.jit
    def forward(
        inp_ptr,
        inp_ch,
        inp_h,
        inp_w,
        inp_bt_st,
        inp_ch_st,
        inp_h_st,
        wgt_ptr,
        wgt_ch,
        wgt_h,
        wgt_w,
        wgt_bt_st,
        wgt_ch_st,
        bis_ptr,
        out_ptr,
        out_ch,
        out_h,
        out_w,
        out_bt_st,
        out_ch_st,
        out_h_st,
        wgt_c_bs: tl.constexpr,
        wgt_h_bs: tl.constexpr,
        wgt_w_bs: tl.constexpr,
        grp_sz: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_grp = language.cdiv(out_w, grp_sz)
        bt = language.batch(pid, out_ch, num_grp, out_w)
        ch = language.channel(pid, out_ch, num_grp, out_w)
        grp = language.row(pid, num_grp, out_w)
        h = grp * grp_sz
        w = language.col(pid, out_w)

        inp_ptr += bt * inp_bt_st + h * inp_h_st + w
        wgt_ptr += ch * wgt_bt_st
        out_ptr += bt * out_bt_st + ch * out_ch_st + h * out_h_st + w

        inp_blk = language.make_conv2d_blk(inp_ch_st, inp_w, wgt_c_bs, wgt_h_bs, wgt_w_bs)
        inp_blk = tl.ravel(inp_blk)
        inp_blk = language.make_group_blk(inp_blk, grp_sz, inp_w)
        inp_msk = language.make_conv2d_msk(inp_ch, inp_h, inp_w, wgt_c_bs, wgt_h_bs, wgt_w_bs)
        inp_msk = tl.ravel(inp_msk)
        inp_msk = language.make_group_msk(inp_msk, grp_sz, h, out_h)
        wgt_blk = language.make_conv2d_blk(wgt_ch_st, wgt_w, wgt_c_bs, wgt_h_bs, wgt_w_bs)
        wgt_blk = tl.ravel(wgt_blk)
        wgt_msk = language.make_conv2d_msk(wgt_ch, wgt_h, wgt_w, wgt_c_bs, wgt_h_bs, wgt_w_bs)
        wgt_msk = tl.ravel(wgt_msk)
        out_blk = tl.arange(0, grp_sz) * out_w
        out_msk = tl.arange(0, grp_sz) + h < out_h

        inp = tl.load(inp_ptr + inp_blk, inp_msk, 0.0)
        wgt = tl.load(wgt_ptr + wgt_blk, wgt_msk, 0.0)
        out = tl.sum(inp * wgt[:, None], 0)

        if bis_ptr:
            out += tl.load(bis_ptr + ch)

        tl.store(out_ptr + out_blk, out, out_msk)
