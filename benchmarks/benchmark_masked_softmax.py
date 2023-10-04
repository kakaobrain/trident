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


def masked_softmax(input: torch.Tensor, mask: torch.Tensor, dim: int):
    input = torch.where(mask.bool(), float("-inf"), input)
    output = torch.nn.functional.softmax(input, dim)

    return output


def build_mask(y_size: int, x_size: int, device=None, dtype=None):
    mask = torch.randint(0, 2, (y_size, x_size), device=device, dtype=dtype)
    mask[0, :] = mask[:, 0] = 0

    return mask


@util.report("masked softmax forward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 16})
def bench_masked_softmax_forward(y_size, x_size, dtype, backend):
    input = torch.randn(y_size, x_size, device="cuda", dtype=dtype)
    mask = build_mask(y_size, x_size, "cuda", dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: masked_softmax(input, mask, 1))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.masked_softmax(input, mask, 1))


@util.report("masked softmax backward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 16})
def bench_masked_softmax_backward(y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs, requires_grad=True)
    mask = build_mask(y_size, x_size, "cuda", dtype)
    grad_output = torch.randn(y_size, x_size, **factory_kwargs)

    if backend == "torch":
        output = masked_softmax(input, mask, 1)
    else:
        output = trident.function.masked_softmax(input, mask, 1)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_masked_softmax_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_masked_softmax_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
