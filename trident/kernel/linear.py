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

from trident import language


def get_configs_linear_io_bound():
    configs = []
    for blk_sz_k in [16, 32, 64]:
        for blk_sz_n in [64, 128]:
            for num_stages in [2, 3]:
                for num_warps in [2, 4]:
                    configs.append(
                        triton.Config(
                            {
                                "blk_sz_m": 64,
                                "blk_sz_k": blk_sz_k,
                                "blk_sz_n": blk_sz_n,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


class Linear:
    @staticmethod
    @triton.autotune(
        configs=get_configs_linear_io_bound(), key=["sz_m", "sz_k", "sz_n"]
    )
    @triton.jit
    def forward(
        x_ptr,
        st_x_m,
        st_x_k,
        y_ptr,
        st_y_m,
        st_y_n,
        w_ptr,
        st_w_n,
        st_w_k,
        b_ptr,
        st_n,
        sz_m,
        sz_k,
        sz_n,
        act: triton.language.constexpr,
        blk_sz_m: triton.language.constexpr,
        blk_sz_k: triton.language.constexpr,
        blk_sz_n: triton.language.constexpr,
    ):
        i = triton.language.program_id(0)
        j = triton.language.program_id(1)

        x_blk_ptr = triton.language.make_block_ptr(
            base=x_ptr,
            shape=(sz_m, sz_k),
            strides=(st_x_m, st_x_k),
            offsets=(i * blk_sz_m, 0),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        w_blk_ptr = triton.language.make_block_ptr(
            base=w_ptr,
            shape=(sz_k, sz_n),
            strides=(st_w_k, st_w_n),
            offsets=(0, j * blk_sz_n),
            block_shape=(blk_sz_k, blk_sz_n),
            order=(1, 0),
        )

        acc = triton.language.zeros((blk_sz_m, blk_sz_n), dtype=triton.language.float32)

        for k in range(0, sz_k, blk_sz_k):
            x = triton.language.load(x_blk_ptr, boundary_check=(0, 1))
            w = triton.language.load(w_blk_ptr, boundary_check=(0, 1))
            acc += triton.language.dot(x, w, False)

            x_blk_ptr = triton.language.advance(x_blk_ptr, (0, blk_sz_k))
            w_blk_ptr = triton.language.advance(w_blk_ptr, (blk_sz_k, 0))

        if b_ptr is not None:
            range_n, msk_n = language.make_block(sz_n, blk_sz_n, j * blk_sz_n)
            b_ptr += range_n * st_n
            b = triton.language.load(b_ptr, msk_n, 0.0)
            acc += b[None, :]

        if act == "relu":
            acc = language.relu(acc)
        elif act == "leaky_relu":
            acc = language.leaky_relu(acc, 1e-2)

        y_blk_ptr = triton.language.make_block_ptr(
            base=y_ptr,
            shape=(sz_m, sz_n),
            strides=(st_y_m, st_y_n),
            offsets=(i * blk_sz_m, j * blk_sz_n),
            block_shape=(blk_sz_m, blk_sz_n),
            order=(1, 0),
        )

        triton.language.store(y_blk_ptr, acc, mask=None, boundary_check=(0, 1))

    @staticmethod
    def backward(grad_out, inp, wgt, out, act):
        grad_act = grad_out
        if act is not None:
            grad_act *= torch.where(out > 0, 1, 0 if act == "relu" else 1e-2)

        grad_inp = grad_act.mm(wgt)
        grad_wgt = grad_act.t().mm(inp)
        grad_bis = grad_act.sum(0)

        return grad_inp, grad_wgt, grad_bis, None
