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


@util.report("argmax forward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 32})
def bench_mean_forward(y_size, x_size, backend):
    input = torch.randn(y_size, x_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.argmax(input, 1))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.argmax(input, 1))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_mean_forward.run(print_data=True, show_plots=show_plots)
    else:
        raise ValueError("The backward isn't supported.")
