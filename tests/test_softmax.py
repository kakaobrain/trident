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


class SoftmaxTestCase(unittest.TestCase):
    m = None
    k = None
    x = None
    t = None

    @classmethod
    def setUpClass(cls):
        cls.m = random.randint(64, 16384)
        cls.k = random.randint(64, 16384)
        cls.x = torch.randn(cls.m, cls.k, device='cuda')
        cls.t = torch.randn(cls.m, cls.k, device='cuda')

    def test_softmax_function(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.softmax(self.x, 1),
            trident.function.softmax(self.x, 1),
        ))

    def test_softmax_operation_forward(self):
        self.assertTrue(triton.testing.allclose(
            torch.nn.functional.softmax(self.x, 1),
            trident.operation.Softmax.apply(self.x, 1),
        ))

    def test_softmax_operation_backward(self):
        criterion = torch.nn.MSELoss()

        def get_torch_grad():
            x = self.x.detach().requires_grad_()
            y = torch.nn.Softmax(dim=1)(x)

            y.retain_grad()
            criterion(y, self.t).backward()

            return y.grad

        def get_trident_grad():
            x = self.x.detach().requires_grad_()
            y = trident.operation.Softmax.apply(x, 1)

            y.retain_grad()
            criterion(y, self.t).backward()

            return y.grad

        self.assertTrue(triton.testing.allclose(
            get_torch_grad(),
            get_trident_grad(),
        ))

    def test_softmax_module(self):
        x = self.x.detach().requires_grad_()
        criterion = torch.nn.MSELoss()

        torch_softmax = torch.nn.Softmax(1)
        trident_softmax = trident.Softmax(1)

        torch_y = torch_softmax(x)
        trident_y = trident_softmax(x)

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
