"""
Copyright 2023 ⓒ Kakao Brain Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
import random
import torch
import torch.nn
import triton
import trident


class LinearTestCase(unittest.TestCase):
    m = None
    k = None
    n = None
    x = None
    w = None
    b = None

    @classmethod
    def setUpClass(cls):
        cls.m = random.randint(64, 2048)
        cls.k = random.randint(64, 2048)
        cls.n = random.randint(64, 2048)
        cls.x = torch.randn(cls.m, cls.k, device='cuda')
        cls.w = torch.randn(cls.n, cls.k, device='cuda')
        cls.b = torch.randn(cls.n, device='cuda')

    def test_linear_no_bias(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.linear(self.x, self.w),
            trident.function.linear(self.x, self.w),
        ))

    def test_linear(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.linear(self.x, self.w, self.b),
            trident.function.linear(self.x, self.w, self.b)
        ))

    def test_linear_zero_input(self):
        x = torch.zeros(self.m, self.k, device='cuda')

        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.linear(x, self.w, self.b),
            trident.function.linear(x, self.w, self.b)
        ))

    def test_linear_zero_bias(self):
        b = torch.zeros(self.n, device='cuda')

        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.linear(self.x, self.w, b),
            trident.function.linear(self.x, self.w, b)
        ))

    def test_linear_relu(self):
        self.assertTrue(triton.testing.allclose(
            torch.relu(torch.nn.functional.linear(self.x, self.w, self.b)),
            trident.function.linear(self.x, self.w, self.b, 'relu')
        ))

    def test_linear_leaky_relu(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.leaky_relu(torch.nn.functional.linear(self.x, self.w, self.b)),
            trident.function.linear(self.x, self.w, self.b, 'leaky_relu')
        ))

    def test_linear_operation_no_bias(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.linear(self.x, self.w),
            trident.operation.Linear.apply(self.x, self.w)
        ))

    def test_linear_operation(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.linear(self.x, self.w, self.b),
            trident.operation.Linear.apply(self.x, self.w, self.b)
        ))

    def test_linear_module_no_bias(self):
        torch_linear = torch.nn.Linear(self.k, self.n, bias=False).to('cuda')
        trident_linear = trident.Linear(self.k, self.n, bias=False)

        trident_linear.load_state_dict(torch_linear.state_dict())

        self.assertTrue(triton.testing.allclose(
            torch_linear(self.x),
            trident_linear(self.x)
        ))

    def test_linear_module(self):
        torch_linear = torch.nn.Linear(self.k, self.n).to('cuda')
        trident_linear = trident.Linear(self.k, self.n)

        trident_linear.load_state_dict(torch_linear.state_dict())

        self.assertTrue(triton.testing.allclose(
            torch_linear(self.x),
            trident_linear(self.x)
        ))


if __name__ == '__main__':
    unittest.main()
