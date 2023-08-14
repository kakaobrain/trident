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
    "cosine similarity forward",
    ["x_size"],
    [256 * i for i in range(1, 21)],
    {"num_batches": 16, "y_size": 16},
)
def bench_cosine_similarity_forward(num_batches, y_size, x_size, ctx):
    factory_kwargs = {"device": "cuda"}

    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs)
    other = torch.randn(num_batches, y_size, x_size, **factory_kwargs)

    if ctx == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.cosine_similarity(input, other, 2))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.cosine_similarity(input, other, 2))


@util.report(
    "cosine similarity backward",
    ["x_size"],
    [256 * i for i in range(1, 21)],
    {"num_batches": 16, "y_size": 16},
)
def bench_cosine_similarity_backward(num_batches, y_size, x_size, ctx):
    factory_kwargs = {"device": "cuda"}

    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs)
    other = torch.randn(num_batches, y_size, x_size, **factory_kwargs)

    input.requires_grad = other.requires_grad = True

    if ctx == "torch":
        operation = torch.nn.CosineSimilarity(2)
    else:
        operation = trident.CosineSimilarity(2)

    output = operation.forward(input, other)
    grad_output = torch.ones_like(output)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_cosine_similarity_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_cosine_similarity_backward.run(print_data=True, show_plots=show_plots)
