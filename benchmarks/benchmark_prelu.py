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
    "prelu forward", ["x_size"], [256 * i for i in range(1, 21)], {"y_size": 16}
)
def bench_prelu_forward(y_size, x_size, ctx):
    factory_kwargs = {"device": "cuda"}

    input = torch.randn(y_size, x_size, **factory_kwargs)
    weight = torch.randn(x_size, **factory_kwargs)

    if ctx == "torch":
        return triton.testing.do_bench(lambda: torch.nn.functional.prelu(input, weight))
    else:
        return triton.testing.do_bench(lambda: trident.function.prelu(input, weight))


@util.report(
    "prelu backward", ["x_size"], [256 * i for i in range(1, 21)], {"y_size": 16}
)
def bench_prelu_backward(y_size, x_size, ctx):
    factory_kwargs = {"device": "cuda"}

    input = torch.randn(y_size, x_size, **factory_kwargs)

    if ctx == "torch":
        operation = torch.nn.PReLU(x_size, 0.3, **factory_kwargs)
    else:
        operation = trident.PReLU(x_size, 0.3, **factory_kwargs)

    output = operation.forward(input)
    grad_output = torch.ones_like(output)

    return triton.testing.do_bench(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_prelu_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_prelu_backward.run(print_data=True, show_plots=show_plots)
