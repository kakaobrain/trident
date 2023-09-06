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


def cosine_similarity_configs():
    configs = []
    for block_size in [256, 512, 1024, 2048]:
        for num_stages in [4, 5]:
            config = triton.Config({"block_size": block_size}, 2 if block_size <= 512 else 4, num_stages)
            configs.append(config)
    return configs


class CosineSimilarity:
    @staticmethod
    @triton.autotune(cosine_similarity_configs(), ["y_size", "x_size"])
    @triton.jit
    def forward(
        output_ptr,
        denominator_ptr,
        numerator_ptr,
        x1_ptr,
        x2_ptr,
        num_batches,
        y_size,
        x_size,
        eps,
        size_along_dim,
        dim: tl.constexpr,
        dtype: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        projected_x_size = y_size if dim == 2 else x_size
        i = pid // projected_x_size
        j = pid % projected_x_size

        if dim == language.dim[0]:
            x1_block_ptr = tl.make_block_ptr(
                x1_ptr,
                shape=(num_batches, y_size, x_size),
                strides=(y_size * x_size, x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(2, 1, 0),
            )
            x2_block_ptr = tl.make_block_ptr(
                x2_ptr,
                shape=(num_batches, y_size, x_size),
                strides=(y_size * x_size, x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(2, 1, 0),
            )
            output_block_ptr = tl.make_block_ptr(
                output_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            denominator_block_ptr = tl.make_block_ptr(
                denominator_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            numerator_block_ptr = tl.make_block_ptr(
                numerator_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
        elif dim == language.dim[1]:
            x1_block_ptr = tl.make_block_ptr(
                x1_ptr,
                shape=(y_size, num_batches, x_size),
                strides=(x_size, y_size * x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(1, 2, 0),
            )
            x2_block_ptr = tl.make_block_ptr(
                x2_ptr,
                shape=(y_size, num_batches, x_size),
                strides=(x_size, y_size * x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(1, 2, 0),
            )
            output_block_ptr = tl.make_block_ptr(
                output_ptr,
                shape=(num_batches, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            denominator_block_ptr = tl.make_block_ptr(
                denominator_ptr,
                shape=(num_batches, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            numerator_block_ptr = tl.make_block_ptr(
                numerator_ptr,
                shape=(num_batches, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
        else:
            x1_block_ptr = tl.make_block_ptr(
                x1_ptr,
                shape=(x_size, num_batches, y_size),
                strides=(1, y_size * x_size, x_size),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(0, 2, 1),
            )
            x2_block_ptr = tl.make_block_ptr(
                x2_ptr,
                shape=(x_size, num_batches, y_size),
                strides=(1, y_size * x_size, x_size),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(0, 2, 1),
            )
            output_block_ptr = tl.make_block_ptr(
                output_ptr,
                shape=(num_batches, y_size),
                strides=(y_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            denominator_block_ptr = tl.make_block_ptr(
                denominator_ptr,
                shape=(num_batches, y_size),
                strides=(y_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            numerator_block_ptr = tl.make_block_ptr(
                numerator_ptr,
                shape=(num_batches, y_size),
                strides=(y_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )

        denominator_accumulation1 = tl.zeros((block_size, 1, 1), tl.float32)
        denominator_accumulation2 = tl.zeros((block_size, 1, 1), tl.float32)
        numerator_accumulation = tl.zeros((block_size, 1, 1), tl.float32)

        for _ in range(0, size_along_dim, block_size):
            x1 = tl.load(x1_block_ptr, boundary_check=(0,), padding_option="zero")
            x2 = tl.load(x2_block_ptr, boundary_check=(0,), padding_option="zero")

            denominator_accumulation1 += x1 * x1
            denominator_accumulation2 += x2 * x2
            numerator_accumulation += x1 * x2

            x1_block_ptr = tl.advance(x1_block_ptr, (block_size, 0, 0))
            x2_block_ptr = tl.advance(x2_block_ptr, (block_size, 0, 0))

        denominator1 = tl.sum(denominator_accumulation1, 0)
        denominator2 = tl.sum(denominator_accumulation2, 0)
        denominator = tl.sqrt(denominator1) * tl.sqrt(denominator2)

        numerator = tl.sum(numerator_accumulation, 0)
        output = numerator / tl.math.max(denominator, eps)

        tl.store(output_block_ptr, output.to(dtype))
        tl.store(denominator_block_ptr, denominator.to(dtype))
        tl.store(numerator_block_ptr, numerator.to(dtype))

    @staticmethod
    @triton.autotune(cosine_similarity_configs(), ["y_size", "x_size"])
    @triton.jit
    def backward(
        grad_x1_ptr,
        grad_x2_ptr,
        grad_output_ptr,
        denominator_ptr,
        numerator_ptr,
        x1_ptr,
        x2_ptr,
        num_batches,
        y_size,
        x_size,
        size_along_dim,
        dim: tl.constexpr,
        dtype: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        projected_x_size = y_size if dim == 2 else x_size
        i = pid // projected_x_size
        j = pid % projected_x_size

        if dim == language.dim[0]:
            grad_x1_block_ptr = tl.make_block_ptr(
                grad_x1_ptr,
                shape=(num_batches, y_size, x_size),
                strides=(y_size * x_size, x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(2, 1, 0),
            )
            grad_x2_block_ptr = tl.make_block_ptr(
                grad_x2_ptr,
                shape=(num_batches, y_size, x_size),
                strides=(y_size * x_size, x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(2, 1, 0),
            )
            grad_output_block_ptr = tl.make_block_ptr(
                grad_output_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            x1_block_ptr = tl.make_block_ptr(
                x1_ptr,
                shape=(num_batches, y_size, x_size),
                strides=(y_size * x_size, x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(2, 1, 0),
            )
            x2_block_ptr = tl.make_block_ptr(
                x2_ptr,
                shape=(num_batches, y_size, x_size),
                strides=(y_size * x_size, x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(2, 1, 0),
            )
            denominator_block_ptr = tl.make_block_ptr(
                denominator_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            numerator_block_ptr = tl.make_block_ptr(
                numerator_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
        elif dim == language.dim[1]:
            grad_x1_block_ptr = tl.make_block_ptr(
                grad_x1_ptr,
                shape=(y_size, num_batches, x_size),
                strides=(x_size, y_size * x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(1, 2, 0),
            )
            grad_x2_block_ptr = tl.make_block_ptr(
                grad_x2_ptr,
                shape=(y_size, num_batches, x_size),
                strides=(x_size, y_size * x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(1, 2, 0),
            )
            grad_output_block_ptr = tl.make_block_ptr(
                grad_output_ptr,
                shape=(num_batches, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            x1_block_ptr = tl.make_block_ptr(
                x1_ptr,
                shape=(y_size, num_batches, x_size),
                strides=(x_size, y_size * x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(1, 2, 0),
            )
            x2_block_ptr = tl.make_block_ptr(
                x2_ptr,
                shape=(y_size, num_batches, x_size),
                strides=(x_size, y_size * x_size, 1),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(1, 2, 0),
            )
            denominator_block_ptr = tl.make_block_ptr(
                denominator_ptr,
                shape=(num_batches, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            numerator_block_ptr = tl.make_block_ptr(
                numerator_ptr,
                shape=(num_batches, x_size),
                strides=(x_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
        else:
            grad_x1_block_ptr = tl.make_block_ptr(
                grad_x1_ptr,
                shape=(x_size, num_batches, y_size),
                strides=(1, y_size * x_size, x_size),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(0, 2, 1),
            )
            grad_x2_block_ptr = tl.make_block_ptr(
                grad_x2_ptr,
                shape=(x_size, num_batches, y_size),
                strides=(1, y_size * x_size, x_size),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(0, 2, 1),
            )
            grad_output_block_ptr = tl.make_block_ptr(
                grad_output_ptr,
                shape=(num_batches, y_size),
                strides=(y_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            x1_block_ptr = tl.make_block_ptr(
                x1_ptr,
                shape=(x_size, num_batches, y_size),
                strides=(1, y_size * x_size, x_size),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(0, 2, 1),
            )
            x2_block_ptr = tl.make_block_ptr(
                x2_ptr,
                shape=(x_size, num_batches, y_size),
                strides=(1, y_size * x_size, x_size),
                offsets=(0, i, j),
                block_shape=(block_size, 1, 1),
                order=(0, 2, 1),
            )
            denominator_block_ptr = tl.make_block_ptr(
                denominator_ptr,
                shape=(num_batches, y_size),
                strides=(y_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )
            numerator_block_ptr = tl.make_block_ptr(
                numerator_ptr,
                shape=(num_batches, y_size),
                strides=(y_size, 1),
                offsets=(i, j),
                block_shape=(1, 1),
                order=(1, 0),
            )

        for _ in range(0, size_along_dim, block_size):
            x1 = tl.load(x1_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

            x2 = tl.load(x2_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

            denominator = tl.load(denominator_block_ptr)
            numerator = tl.load(numerator_block_ptr)
            grad_output = tl.load(grad_output_block_ptr)

            square_sum1 = tl.sum(x1 * x1, 0)
            square_sum2 = tl.sum(x2 * x2, 0)

            grad_denominator = grad_output * numerator * (-1 / (denominator * denominator))

            grad_mul1 = grad_denominator * language.distance(x2, 0)
            grad_mul2 = grad_denominator * language.distance(x1, 0)

            grad_sqrt1 = grad_mul1 / (2 * tl.sqrt(square_sum1))
            grad_sqrt2 = grad_mul2 / (2 * tl.sqrt(square_sum2))

            grad_to_dot = grad_output / denominator

            grad_x1 = (grad_sqrt1 * 2 * x1) + (grad_to_dot * x2)
            grad_x2 = (grad_sqrt2 * 2 * x2) + (grad_to_dot * x1)

            tl.store(
                grad_x1_block_ptr,
                grad_x1.to(dtype),
                mask=None,
                boundary_check=(0,),
            )
            tl.store(
                grad_x2_block_ptr,
                grad_x2.to(dtype),
                mask=None,
                boundary_check=(0,),
            )

            x1_block_ptr = tl.advance(x1_block_ptr, (block_size, 0, 0))
            x2_block_ptr = tl.advance(x2_block_ptr, (block_size, 0, 0))
            grad_x1_block_ptr = tl.advance(grad_x1_block_ptr, (block_size, 0, 0))
            grad_x2_block_ptr = tl.advance(grad_x2_block_ptr, (block_size, 0, 0))
