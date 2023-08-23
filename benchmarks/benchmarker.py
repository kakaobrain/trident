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

import argparse

import benchmark_adaptive_avg_pool2d
import benchmark_argmax
import benchmark_attention
import benchmark_batch_norm
import benchmark_conv2d
import benchmark_cosine_similarity
import benchmark_dropout
import benchmark_geglu
import benchmark_gelu
import benchmark_group_norm
import benchmark_instance_norm
import benchmark_layer_norm
import benchmark_leaky_relu
import benchmark_linear
import benchmark_max
import benchmark_max_pool2d
import benchmark_mean
import benchmark_prelu
import benchmark_relu
import benchmark_silu
import benchmark_softmax
import benchmark_sum
import benchmark_var
import torch


def print_scenarios():
    print(f"Following scenarios can be chosen:")
    print(
        ", ".join(
            [
                "adaptive-avg-pool2d",
                "argmax",
                "attention",
                "batch-norm",
                "conv2d",
                "cosine-similarity",
                "dropout",
                "geglu",
                "gelu",
                "group-norm",
                "instance-norm",
                "layer-norm",
                "leaky-relu",
                "linear",
                "max",
                "max-pool2d",
                "mean",
                "prelu",
                "relu",
                "silu",
                "softmax",
                "sum",
                "var",
            ]
        )
    )


def run_benchmarks(scenario, mode, show_plots):
    if scenario == "adaptive-avg-pool2d":
        benchmark_adaptive_avg_pool2d.run_benchmark(mode, show_plots)
    elif scenario == "argmax":
        benchmark_argmax.run_benchmark(mode, show_plots)
    elif scenario == "attention":
        benchmark_attention.run_benchmark(mode, show_plots)
    elif scenario == "batch-norm":
        benchmark_batch_norm.run_benchmark(mode, show_plots)
    elif scenario == "conv2d":
        benchmark_conv2d.run_benchmark(mode, show_plots)
    elif scenario == "cosine-similarity":
        benchmark_cosine_similarity.run_benchmark(mode, show_plots)
    elif scenario == "dropout":
        benchmark_dropout.run_benchmark(mode, show_plots)
    elif scenario == "geglu":
        benchmark_geglu.run_benchmark(mode, show_plots)
    elif scenario == "gelu":
        benchmark_gelu.run_benchmark(mode, show_plots)
    elif scenario == "group-norm":
        benchmark_group_norm.run_benchmark(mode, show_plots)
    elif scenario == "instance-norm":
        benchmark_instance_norm.run_benchmark(mode, show_plots)
    elif scenario == "layer-norm":
        benchmark_layer_norm.run_benchmark(mode, show_plots)
    elif scenario == "leaky-relu":
        benchmark_leaky_relu.run_benchmark(mode, show_plots)
    elif scenario == "linear":
        benchmark_linear.run_benchmark(mode, show_plots)
    elif scenario == "max":
        benchmark_max.run_benchmark(mode, show_plots)
    elif scenario == "max-pool2d":
        benchmark_max_pool2d.run_benchmark(mode, show_plots)
    elif scenario == "mean":
        benchmark_mean.run_benchmark(mode, show_plots)
    elif scenario == "prelu":
        benchmark_prelu.run_benchmark(mode, show_plots)
    elif scenario == "relu":
        benchmark_relu.run_benchmark(mode, show_plots)
    elif scenario == "silu":
        benchmark_silu.run_benchmark(mode, show_plots)
    elif scenario == "softmax":
        benchmark_softmax.run_benchmark(mode, show_plots)
    elif scenario == "sum":
        benchmark_sum.run_benchmark(mode, show_plots)
    elif scenario == "var":
        benchmark_var.run_benchmark(mode, show_plots)
    elif not scenario:
        benchmark_adaptive_avg_pool2d.run_benchmark(mode, show_plots)
        benchmark_argmax.run_benchmark(mode, show_plots)
        benchmark_attention.run_benchmark(mode, show_plots)
        benchmark_batch_norm.run_benchmark(mode, show_plots)
        benchmark_conv2d.run_benchmark(mode, show_plots)
        benchmark_cosine_similarity.run_benchmark(mode, show_plots)
        benchmark_dropout.run_benchmark(mode, show_plots)
        benchmark_geglu.run_benchmark(mode, show_plots)
        benchmark_gelu.run_benchmark(mode, show_plots)
        benchmark_group_norm.run_benchmark(mode, show_plots)
        benchmark_instance_norm.run_benchmark(mode, show_plots)
        benchmark_layer_norm.run_benchmark(mode, show_plots)
        benchmark_leaky_relu.run_benchmark(mode, show_plots)
        benchmark_linear.run_benchmark(mode, show_plots)
        benchmark_max.run_benchmark(mode, show_plots)
        benchmark_max_pool2d.run_benchmark(mode, show_plots)
        benchmark_mean.run_benchmark(mode, show_plots)
        benchmark_prelu.run_benchmark(mode, show_plots)
        benchmark_relu.run_benchmark(mode, show_plots)
        benchmark_silu.run_benchmark(mode, show_plots)
        benchmark_softmax.run_benchmark(mode, show_plots)
        benchmark_sum.run_benchmark(mode, show_plots)
        benchmark_var.run_benchmark(mode, show_plots)
    else:
        print_scenarios()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="specify a scenario to run", type=str)
    parser.add_argument(
        "--mode",
        choices=["forward", "backward"],
        default="forward",
        help="specify a mode to run",
        type=str,
    )
    parser.add_argument("--list", action="store_true", help="list all scenarios can be run")
    parser.add_argument("--show-plots", action="store_true", help="show plots")
    args = parser.parse_args()

    if args.list:
        print_scenarios()
    else:
        torch.cuda.set_stream(torch.cuda.Stream())
        run_benchmarks(
            args.scenario.replace("_", "-") if args.scenario else None,
            args.mode,
            args.show_plots,
        )


if __name__ == "__main__":
    main()
