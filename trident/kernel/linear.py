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


def get_configs_linear_io_bound_forward():
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


def get_configs_linear_io_bound_backward():
    configs = []
    for blk_sz_k in [32, 64]:
        for blk_sz_n_m in [32, 64]:
            for num_stages in [2, 3, 5]:
                for num_warps in [2, 4, 8]:
                    configs.append(
                        triton.Config(
                            {
                                "blk_sz_m": blk_sz_n_m,
                                "blk_sz_k": blk_sz_k,
                                "blk_sz_n": blk_sz_n_m,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


class Linear:
    @staticmethod
    @triton.autotune(
        configs=get_configs_linear_io_bound_forward(), key=["sz_m", "sz_k", "sz_n"]
    )
    @triton.jit
    def forward(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        st_n,
        sz_m,
        sz_k,
        sz_n,
        act: triton.language.constexpr,
        blk_sz_m: triton.language.constexpr,
        blk_sz_k: triton.language.constexpr,
        blk_sz_n: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        i = triton.language.program_id(0)
        j = triton.language.program_id(1)

        x_blk_ptr = triton.language.make_block_ptr(
            base=x_ptr,
            shape=(sz_m, sz_k),
            strides=(sz_k, 1),
            offsets=(i * blk_sz_m, 0),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        w_blk_ptr = triton.language.make_block_ptr(
            base=w_ptr,
            shape=(sz_k, sz_n),
            strides=(1, sz_k),
            offsets=(0, j * blk_sz_n),
            block_shape=(blk_sz_k, blk_sz_n),
            order=(1, 0),
        )

        acc = triton.language.zeros((blk_sz_m, blk_sz_n), dtype=dtype)

        for _ in range(0, sz_k, blk_sz_k):
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
            strides=(sz_n, 1),
            offsets=(i * blk_sz_m, j * blk_sz_n),
            block_shape=(blk_sz_m, blk_sz_n),
            order=(1, 0),
        )

        triton.language.store(y_blk_ptr, acc, mask=None, boundary_check=(0, 1))

    @staticmethod
    @triton.jit
    def backward_bias(
        p_grad_out,
        p_out,
        p_grad_act,
        p_grad_bis,
        sz_m,
        sz_n,
        act: triton.language.constexpr,
        blk_sz: triton.language.constexpr,
    ):
        n = triton.language.program_id(0)

        ptrs_grad_out = triton.language.make_block_ptr(
            base=p_grad_out,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(0, n),
            block_shape=(blk_sz, 1),
            order=(1, 0),
        )

        ptrs_out = triton.language.make_block_ptr(
            base=p_out,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(0, n),
            block_shape=(blk_sz, 1),
            order=(1, 0),
        )

        ptrs_grad_act = triton.language.make_block_ptr(
            base=p_grad_act,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(0, n),
            block_shape=(blk_sz, 1),
            order=(1, 0),
        )

        acc_grad_bis = 0.0

        for m in range(0, triton.language.cdiv(sz_m, blk_sz)):
            grad_act = triton.language.load(ptrs_grad_out, boundary_check=(0,))
            out = triton.language.load(ptrs_out, boundary_check=(0,))

            if act is not None:
                grad_act *= triton.language.where(
                    out > 0, 1, 0 if act == "relu" else 1e-2
                )

            triton.language.store(
                ptrs_grad_act, grad_act, mask=None, boundary_check=(0,)
            )

            if p_grad_bis is not None:
                acc_grad_bis += language.sum(grad_act)

            ptrs_grad_out = triton.language.advance(ptrs_grad_out, (blk_sz, 0))
            ptrs_out = triton.language.advance(ptrs_out, (blk_sz, 0))
            ptrs_grad_act = triton.language.advance(ptrs_grad_act, (blk_sz, 0))

        if p_grad_bis is not None:
            triton.language.store(p_grad_bis + n, acc_grad_bis)

    @staticmethod
    @triton.autotune(
        configs=get_configs_linear_io_bound_backward(), key=["sz_m", "sz_k", "sz_n"]
    )
    @triton.jit
    def backward(
        p_grad_out,
        p_wgt,
        p_inp,
        p_grad_inp,
        p_grad_wgt,
        sz_m,
        sz_n,
        sz_k,
        blk_sz_m: triton.language.constexpr,
        blk_sz_k: triton.language.constexpr,
        blk_sz_n: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        i = triton.language.program_id(0)
        j = triton.language.program_id(1)

        ptrs_grad_out = triton.language.make_block_ptr(
            base=p_grad_out,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(i * blk_sz_m, 0),
            block_shape=(blk_sz_m, blk_sz_n),
            order=(1, 0),
        )

        ptrs_wgt = triton.language.make_block_ptr(
            base=p_wgt,
            shape=(sz_n, sz_k),
            strides=(sz_k, 1),
            offsets=(0, j * blk_sz_k),
            block_shape=(blk_sz_n, blk_sz_k),
            order=(1, 0),
        )

        acc_mk = triton.language.zeros((blk_sz_m, blk_sz_k), dtype)

        for _ in range(0, sz_n, blk_sz_n):
            grad = triton.language.load(ptrs_grad_out, boundary_check=(0, 1))
            wgt = triton.language.load(ptrs_wgt, boundary_check=(0, 1))
            acc_mk += triton.language.dot(grad, wgt, False)

            ptrs_grad_out = triton.language.advance(ptrs_grad_out, (0, blk_sz_n))
            ptrs_wgt = triton.language.advance(ptrs_wgt, (blk_sz_n, 0))

        ptrs_grad_inp = triton.language.make_block_ptr(
            base=p_grad_inp,
            shape=(sz_m, sz_k),
            strides=(sz_k, 1),
            offsets=(i * blk_sz_m, j * blk_sz_k),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        triton.language.store(ptrs_grad_inp, acc_mk, mask=None, boundary_check=(0, 1))

        ptrs_grad_out_t = triton.language.make_block_ptr(
            base=p_grad_out,
            shape=(sz_n, sz_m),
            strides=(1, sz_n),
            offsets=(i * blk_sz_n, 0),
            block_shape=(blk_sz_n, blk_sz_m),
            order=(1, 0),
        )

        ptrs_inp = triton.language.make_block_ptr(
            base=p_inp,
            shape=(sz_m, sz_k),
            strides=(sz_k, 1),
            offsets=(0, j * blk_sz_k),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        acc_nk = triton.language.zeros((blk_sz_n, blk_sz_k), dtype)

        for _ in range(0, sz_m, blk_sz_m):
            grad_t = triton.language.load(ptrs_grad_out_t, boundary_check=(0, 1))
            inp = triton.language.load(ptrs_inp, boundary_check=(0, 1))
            acc_nk += triton.language.dot(grad_t, inp, False)

            ptrs_grad_out_t = triton.language.advance(ptrs_grad_out_t, (0, blk_sz_m))
            ptrs_inp = triton.language.advance(ptrs_inp, (blk_sz_m, 0))

        ptrs_grad_wgt = triton.language.make_block_ptr(
            base=p_grad_wgt,
            shape=(sz_n, sz_k),
            strides=(sz_k, 1),
            offsets=(i * blk_sz_n, j * blk_sz_k),
            block_shape=(blk_sz_n, blk_sz_k),
            order=(1, 0),
        )

        triton.language.store(ptrs_grad_wgt, acc_nk, mask=None, boundary_check=(0, 1))
