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
import benchmark_conv2d
import benchmark_instance_norm
import benchmark_leaky_relu
import benchmark_linear
import benchmark_max_pool2d
import benchmark_relu
import benchmark_softmax


def print_scenarios():
    print(f'Following scenarios can be chosen:')
    print(', '.join([
        'adaptive-avg-pool2d',
        'conv2d',
        'instance-norm',
        'leaky-relu',
        'linear',
        'max-pool2d',
        'relu',
        'softmax'
    ]))


def run_benchmarks(args):
    if args.scenario == 'adaptive-avg-pool2d':
        benchmark_adaptive_avg_pool2d.run_benchmarks(args.show_plots)
    elif args.scenario == 'conv2d':
        benchmark_conv2d.run_benchmarks(args.show_plots)
    elif args.scenario == 'instance-norm':
        benchmark_instance_norm.run_benchmarks(args.show_plots)
    elif args.scenario == 'leaky-relu':
        benchmark_leaky_relu.run_benchmarks(args.show_plots)
    elif args.scenario == 'linear':
        benchmark_linear.run_benchmarks(args.show_plots)
    elif args.scenario == 'max-pool2d':
        benchmark_max_pool2d.run_benchmarks(args.show_plots)
    elif args.scenario == 'relu':
        benchmark_relu.run_benchmarks(args.show_plots)
    elif args.scenario == 'softmax':
        benchmark_softmax.run_benchmarks(args.show_plots)
    elif not args.scenario:
        benchmark_instance_norm.run_benchmarks(args.show_plots)
        benchmark_conv2d.run_benchmarks(args.show_plots)
        benchmark_leaky_relu.run_benchmarks(args.show_plots)
        benchmark_linear.run_benchmarks(args.show_plots)
        benchmark_max_pool2d.run_benchmarks(args.show_plots)
        benchmark_relu.run_benchmarks(args.show_plots)
        benchmark_softmax.run_benchmarks(args.show_plots)
    else:
        print_scenarios()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, help='specify a scenario to run')
    parser.add_argument('--list', action='store_true', help='list all scenarios can be run')
    parser.add_argument('--show-plots', action='store_true', help='show plots')

    args = parser.parse_args()

    if args.list:
        print_scenarios()
    else:
        run_benchmarks(args)


if __name__ == '__main__':
    main()
