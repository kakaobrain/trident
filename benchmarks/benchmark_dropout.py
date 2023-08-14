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


@util.report("dropout forward", ["inp_sz"], [32 * i for i in range(1, 21)], {"p": 0.5})
def bench_dropout_forward(p, inp_sz, backend):
    inp = torch.randn(inp_sz, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.dropout(inp, p))
    else:
        return triton.testing.do_bench(lambda: trident.function.dropout(inp, p))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_dropout_forward.run(print_data=True, show_plots=show_plots)
    else:
        raise NotImplementedError("The backward isn't implemented.")
