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
    @triton.autotune(configs=get_configs_linear_io_bound_forward(), key=["sz_m", "sz_k", "sz_n"])
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
        act: tl.constexpr,
        blk_sz_m: tl.constexpr,
        blk_sz_k: tl.constexpr,
        blk_sz_n: tl.constexpr,
        dtype: tl.constexpr,
    ):
        i = tl.program_id(0)
        j = tl.program_id(1)

        x_blk_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(sz_m, sz_k),
            strides=(sz_k, 1),
            offsets=(i * blk_sz_m, 0),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        w_blk_ptr = tl.make_block_ptr(
            base=w_ptr,
            shape=(sz_k, sz_n),
            strides=(1, sz_k),
            offsets=(0, j * blk_sz_n),
            block_shape=(blk_sz_k, blk_sz_n),
            order=(1, 0),
        )

        acc = tl.zeros((blk_sz_m, blk_sz_n), dtype=dtype)

        for _ in range(0, sz_k, blk_sz_k):
            x = tl.load(x_blk_ptr, boundary_check=(0, 1))
            w = tl.load(w_blk_ptr, boundary_check=(0, 1))
            acc += tl.dot(x, w, False)

            x_blk_ptr = tl.advance(x_blk_ptr, (0, blk_sz_k))
            w_blk_ptr = tl.advance(w_blk_ptr, (blk_sz_k, 0))

        if b_ptr is not None:
            range_n, msk_n = language.make_block(sz_n, blk_sz_n, j * blk_sz_n)
            b_ptr += range_n * st_n
            b = tl.load(b_ptr, msk_n, 0.0)
            acc += b[None, :]

        if act == "relu":
            acc = language.relu(acc)
        elif act == "leaky_relu":
            acc = language.leaky_relu(acc, 1e-2)

        y_blk_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(i * blk_sz_m, j * blk_sz_n),
            block_shape=(blk_sz_m, blk_sz_n),
            order=(1, 0),
        )

        tl.store(y_blk_ptr, acc, mask=None, boundary_check=(0, 1))

    @staticmethod
    @triton.jit
    def backward_bias(
        p_grad_out,
        p_out,
        p_grad_act,
        p_grad_bis,
        sz_m,
        sz_n,
        act: tl.constexpr,
        blk_sz: tl.constexpr,
    ):
        n = tl.program_id(0)

        ptrs_grad_out = tl.make_block_ptr(
            base=p_grad_out,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(0, n),
            block_shape=(blk_sz, 1),
            order=(1, 0),
        )

        ptrs_out = tl.make_block_ptr(
            base=p_out,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(0, n),
            block_shape=(blk_sz, 1),
            order=(1, 0),
        )

        ptrs_grad_act = tl.make_block_ptr(
            base=p_grad_act,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(0, n),
            block_shape=(blk_sz, 1),
            order=(1, 0),
        )

        acc_grad_bis = 0.0

        for m in range(0, tl.cdiv(sz_m, blk_sz)):
            grad_act = tl.load(ptrs_grad_out, boundary_check=(0,))
            out = tl.load(ptrs_out, boundary_check=(0,))

            if act is not None:
                grad_act *= tl.where(out > 0, 1, 0 if act == "relu" else 1e-2)

            tl.store(ptrs_grad_act, grad_act, mask=None, boundary_check=(0,))

            if p_grad_bis is not None:
                acc_grad_bis += tl.sum(tl.ravel(grad_act), 0)

            ptrs_grad_out = tl.advance(ptrs_grad_out, (blk_sz, 0))
            ptrs_out = tl.advance(ptrs_out, (blk_sz, 0))
            ptrs_grad_act = tl.advance(ptrs_grad_act, (blk_sz, 0))

        if p_grad_bis is not None:
            tl.store(p_grad_bis + n, acc_grad_bis)

    @staticmethod
    @triton.autotune(configs=get_configs_linear_io_bound_backward(), key=["sz_m", "sz_k", "sz_n"])
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
        blk_sz_m: tl.constexpr,
        blk_sz_k: tl.constexpr,
        blk_sz_n: tl.constexpr,
        dtype: tl.constexpr,
    ):
        i = tl.program_id(0)
        j = tl.program_id(1)

        ptrs_grad_out = tl.make_block_ptr(
            base=p_grad_out,
            shape=(sz_m, sz_n),
            strides=(sz_n, 1),
            offsets=(i * blk_sz_m, 0),
            block_shape=(blk_sz_m, blk_sz_n),
            order=(1, 0),
        )

        ptrs_wgt = tl.make_block_ptr(
            base=p_wgt,
            shape=(sz_n, sz_k),
            strides=(sz_k, 1),
            offsets=(0, j * blk_sz_k),
            block_shape=(blk_sz_n, blk_sz_k),
            order=(1, 0),
        )

        acc_mk = tl.zeros((blk_sz_m, blk_sz_k), dtype)

        for _ in range(0, sz_n, blk_sz_n):
            grad = tl.load(ptrs_grad_out, boundary_check=(0, 1))
            wgt = tl.load(ptrs_wgt, boundary_check=(0, 1))
            acc_mk += tl.dot(grad, wgt, False)

            ptrs_grad_out = tl.advance(ptrs_grad_out, (0, blk_sz_n))
            ptrs_wgt = tl.advance(ptrs_wgt, (blk_sz_n, 0))

        ptrs_grad_inp = tl.make_block_ptr(
            base=p_grad_inp,
            shape=(sz_m, sz_k),
            strides=(sz_k, 1),
            offsets=(i * blk_sz_m, j * blk_sz_k),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        tl.store(ptrs_grad_inp, acc_mk, mask=None, boundary_check=(0, 1))

        ptrs_grad_out_t = tl.make_block_ptr(
            base=p_grad_out,
            shape=(sz_n, sz_m),
            strides=(1, sz_n),
            offsets=(i * blk_sz_n, 0),
            block_shape=(blk_sz_n, blk_sz_m),
            order=(1, 0),
        )

        ptrs_inp = tl.make_block_ptr(
            base=p_inp,
            shape=(sz_m, sz_k),
            strides=(sz_k, 1),
            offsets=(0, j * blk_sz_k),
            block_shape=(blk_sz_m, blk_sz_k),
            order=(1, 0),
        )

        acc_nk = tl.zeros((blk_sz_n, blk_sz_k), dtype)

        for _ in range(0, sz_m, blk_sz_m):
            grad_t = tl.load(ptrs_grad_out_t, boundary_check=(0, 1))
            inp = tl.load(ptrs_inp, boundary_check=(0, 1))
            acc_nk += tl.dot(grad_t, inp, False)

            ptrs_grad_out_t = tl.advance(ptrs_grad_out_t, (0, blk_sz_m))
            ptrs_inp = tl.advance(ptrs_inp, (blk_sz_m, 0))

        ptrs_grad_wgt = tl.make_block_ptr(
            base=p_grad_wgt,
            shape=(sz_n, sz_k),
            strides=(sz_k, 1),
            offsets=(i * blk_sz_n, j * blk_sz_k),
            block_shape=(blk_sz_n, blk_sz_k),
            order=(1, 0),
        )

        tl.store(ptrs_grad_wgt, acc_nk, mask=None, boundary_check=(0, 1))
