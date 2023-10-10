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
import util

import trident


@util.report(
    "attention forward", ["y_size"], [2**i for i in range(5, 10)], {"num_batches": 64, "num_heads": 8, "x_size": 64}
)
def bench_attention_forward(num_batches, num_heads, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    query = torch.randn(num_batches, num_heads, y_size, x_size, **factory_kwargs)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    if backend == "torch":
        return triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(query, key, value))
    else:
        return triton.testing.do_bench(
            lambda: trident.function.scaled_dot_product_attention(query, key, value, use_accelerator=True)
        )


@util.report(
    "attention backward",
    ["y_size"],
    [2**i for i in range(5, 10)],
    {"num_batches": 32, "num_heads": 8, "x_size": 64},
)
def bench_attention_backward(num_batches, num_heads, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    query = torch.randn(num_batches, num_heads, y_size, x_size, **factory_kwargs, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    grad_output = torch.randn(num_batches, num_heads, y_size, x_size, **factory_kwargs)

    if backend == "torch":
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    else:
        output = trident.function.scaled_dot_product_attention(query, key, value, use_accelerator=True)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_attention_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_attention_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
