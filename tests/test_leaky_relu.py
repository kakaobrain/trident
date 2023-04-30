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
import triton
import trident


class LeakyReLUTestCase(unittest.TestCase):
    m = None
    n = None
    x = None
    t = None

    @classmethod
    def setUpClass(cls):
        cls.m = random.randint(64, 5120)
        cls.n = random.randint(64, 5120)
        cls.x = torch.randn(cls.m, cls.n, device='cuda')
        cls.t = torch.randn(cls.m, cls.n, device='cuda')

    def test_leaky_relu_function(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.leaky_relu(self.x),
            trident.function.leaky_relu(self.x),
        ))

    def test_leaky_relu_operation_forward(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.leaky_relu(self.x, 1e-2),
            trident.operation.LeakyReLU.apply(self.x, 1e-2),
        ))

    def test_leaky_relu_operation_backward(self):
        criterion = torch.nn.MSELoss()

        def get_torch_grad():
            x = self.x.detach().requires_grad_()
            y = torch.nn.LeakyReLU(1e-2)(x)

            y.retain_grad()
            criterion(y, self.t).backward()

            return y.grad

        def get_trident_grad():
            x = self.x.detach().requires_grad_()
            y = trident.operation.LeakyReLU.apply(x, 1e-2)

            y.retain_grad()
            criterion(y, self.t).backward()

            return y.grad

        self.assertTrue(triton.testing.allclose(
            get_torch_grad(),
            get_trident_grad(),
        ))

    def test_leaky_relu_module(self):
        x = self.x.detach().requires_grad_()
        criterion = torch.nn.MSELoss()

        torch_leaky_relu = torch.nn.LeakyReLU()
        trident_leaky_relu = trident.LeakyReLU()

        torch_y = torch_leaky_relu(x)
        trident_y = trident_leaky_relu(x)

        self.assertTrue(triton.testing.allclose(
            torch_y,
            trident_y,
        ))

        torch_y.retain_grad()
        trident_y.retain_grad()

        criterion(torch_y, self.t).backward()
        criterion(trident_y, self.t).backward()

        self.assertTrue(triton.testing.allclose(
            torch_y.grad,
            trident_y.grad,
        ))


if __name__ == '__main__':
    unittest.main()
