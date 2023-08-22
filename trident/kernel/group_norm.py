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


def configs_for_forward():
    configs = []
    for num_stages in [2, 3, 4]:
        for num_warps in [4, 8]:
            for x_block_size in [128, 256, 512]:
                config = triton.Config(
                    {"x_block_size": x_block_size},
                    num_stages=num_stages,
                    num_warps=num_warps,
                )
                configs.append(config)
    return configs


class GroupNorm:
    @staticmethod
    @triton.autotune(
        configs=configs_for_forward(),
        key=["x_size"],
    )
    @triton.jit
    def forward(
        output_ptr,
        input_ptr,
        y_size,
        x_size,
        num_groups,
        weight_ptr,
        bias_ptr,
        eps,
        x_block_size: tl.constexpr,
        group_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        group_size = y_size // num_groups
        pid = tl.program_id(0)
        batch = pid // num_groups
        group = pid % num_groups
        batch_offset = batch * y_size * x_size
        group_offset = group * group_size * x_size

        mean = language.Mean.forward(
            input_ptr + batch_offset,
            num_groups,
            x_size * group_size,
            x_size * group_size,
            1,
            group,
            dtype,
            x_block_size,
        )
        var = language.var(
            input_ptr + batch_offset,
            num_groups,
            x_size * group_size,
            group,
            mean,
            language.dim[1],
            language.zero,
            x_block_size,
            dtype,
        )
        std = language.std(var, eps)

        input_block_ptr = tl.make_block_ptr(
            input_ptr + batch_offset + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr + batch_offset + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(y_size, 1),
                strides=(1, y_size),
                offsets=(group * group_size, 0),
                block_shape=(group_block_size, 1),
                order=(0, 1),
            )
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
        else:
            weight = None

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(y_size, 1),
                strides=(1, y_size),
                offsets=(group * group_size, 0),
                block_shape=(group_block_size, 1),
                order=(0, 1),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(0,))
        else:
            bias = None

        for _ in range(0, x_size, x_block_size):
            input = tl.load(input_block_ptr, boundary_check=(0, 1))
            output = language.norm(input, mean, std)

            if weight is not None:
                output *= weight

            if bias is not None:
                output += bias

            tl.store(output_block_ptr, output.to(dtype), boundary_check=(0, 1))
            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))
            output_block_ptr = tl.advance(output_block_ptr, (0, x_block_size))

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr,
        grad_weight_staging_ptr,
        grad_bias_staging_ptr,
        grad_output_ptr,
        input_ptr,
        y_size,
        x_size,
        num_groups,
        weight_ptr,
        eps,
        x_block_size: tl.constexpr,
        group_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        group_size = y_size // num_groups
        pid = tl.program_id(0)
        batch = pid // num_groups
        group = pid % num_groups
        batch_offset = batch * y_size * x_size
        group_offset = group * group_size * x_size

        mean = language.Mean.forward(
            input_ptr + batch_offset,
            num_groups,
            x_size * group_size,
            x_size * group_size,
            1,
            group,
            dtype,
            x_block_size,
        )
        var = language.var(
            input_ptr + batch_offset,
            num_groups,
            x_size * group_size,
            group,
            mean,
            language.dim[1],
            language.zero,
            x_block_size,
            dtype,
        )
        std = language.std(var, eps)

        input_block_ptr = tl.make_block_ptr(
            input_ptr + batch_offset + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        input = tl.load(input_block_ptr, boundary_check=(0, 1))
        y_condition = tl.arange(0, group_block_size) < group_size
        x_condition = tl.arange(0, x_block_size) < x_size
        condition = y_condition[:, None] & x_condition[None, :]
        centered_mean = tl.where(condition, input - mean, 0)

        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr + batch_offset + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1))

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(y_size, 1),
                strides=(1, y_size),
                offsets=(group * group_size, 0),
                block_shape=(group_block_size, 1),
                order=(0, 1),
            )
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
            grad_norm = weight * grad_output
        else:
            grad_norm = grad_output

        grad_std = tl.sum(grad_norm * centered_mean, 1)
        grad_std = tl.sum(grad_std, 0)
        grad_var = grad_std / (-language.pow2(std) * 2 * std * x_size * group_size)
        grad_distance = 2 * centered_mean * grad_var
        grad_centered_mean = tl.where(condition, (grad_norm / std) + grad_distance, 0)
        grad_mean = tl.sum(grad_centered_mean, 1) / x_size
        grad_mean = -tl.sum(grad_mean, 0) / group_size
        grad_input = grad_centered_mean + grad_mean

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr + batch_offset + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0, 1))

        if grad_weight_staging_ptr is not None:
            norm = centered_mean / std
            grad_weight = tl.sum(norm * grad_output, 1)
            grad_weight_staging_block_ptr = tl.make_block_ptr(
                grad_weight_staging_ptr + batch * y_size + group * group_size,
                shape=(group_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(group_block_size,),
                order=(0,),
            )
            tl.store(
                grad_weight_staging_block_ptr,
                grad_weight.to(dtype),
                boundary_check=(0,),
            )

        if grad_bias_staging_ptr is not None:
            grad_bias = tl.sum(grad_output, 1)
            grad_bias_staging_block_ptr = tl.make_block_ptr(
                grad_bias_staging_ptr + batch * y_size + group * group_size,
                shape=(group_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(group_block_size,),
                order=(0,),
            )
            tl.store(grad_bias_staging_block_ptr, grad_bias.to(dtype), boundary_check=(0,))
