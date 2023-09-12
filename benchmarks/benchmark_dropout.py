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


@util.report("dropout forward", ["x_size"], [128 * i for i in range(1, 21)], {"p": 0.5})
def bench_dropout_forward(x_size, p, dtype, backend):
    input = torch.randn(x_size, device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.dropout(input, p))
    else:
        return triton.testing.do_bench(lambda: trident.function.dropout(input, p))


@util.report("dropout backward", ["x_size"], [128 * i for i in range(1, 21)], {"p": 0.5})
def bench_dropout_backward(x_size, p, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.randn(x_size, **factory_kwargs)

    if backend == "torch":
        output = torch.nn.functional.dropout(input, p)
    else:
        output = trident.function.dropout(input, p)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_dropout_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_dropout_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
