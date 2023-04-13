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
import torch
import torch.nn.functional as tf
import triton
import triton.testing as tt
import trident


class LinearTestCase(unittest.TestCase):
    def test_small_tensor_no_bias(self):
        x = torch.randn(512, 512, device='cuda')
        w = torch.randn(256, 512, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w), tf.linear(x, w)))

    def test_large_tensor_no_bias(self):
        x = torch.randn(1024, 2048, device='cuda')
        w = torch.randn(4096, 2048, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w), tf.linear(x, w)))

    def test_small_tensor(self):
        x = torch.randn(64, 32, device='cuda')
        w = torch.randn(16, 32, device='cuda')
        b = torch.randn(16, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w, b), tf.linear(x, w, b)))

    def test_large_tensor(self):
        x = torch.randn(2048, 4096, device='cuda')
        w = torch.randn(1024, 4096, device='cuda')
        b = torch.randn(1024, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w, b), tf.linear(x, w, b)))

    def test_linear_zero_input(self):
        x = torch.full((128, 32), 0.0, device='cuda')
        w = torch.randn(256, 32, device='cuda')
        b = torch.full((1, 256), 0.1, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w, b), tf.linear(x, w, b)))

    def test_linear_zero_bias(self):
        x = torch.randn(128, 32, device='cuda')
        w = torch.randn(256, 32, device='cuda')
        b = torch.full((1, 256), 0.0, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w, b), tf.linear(x, w, b)))

    def test_linear_relu(self):
        x = torch.randn(1, 32, device='cuda')
        w = torch.randn(8, 32, device='cuda')
        b = torch.randn(8, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w, b, 'relu'),
                                    torch.relu(tf.linear(x, w, b))))

    def test_linear_leaky_relu(self):
        x = torch.randn(1, 32, device='cuda')
        w = torch.randn(8, 32, device='cuda')
        b = torch.randn(8, device='cuda')

        self.assertTrue(tt.allclose(trident.function.linear(x, w, b, 'leaky_relu'),
                                    tf.leaky_relu(tf.linear(x, w, b))))

    def test_linear_operation_no_bias(self):
        x = torch.randn(1024, 4096, device='cuda')
        w = torch.randn(2048, 4096, device='cuda')

        self.assertTrue(
            triton.testing.allclose(trident.operation.Linear.apply(x, w), torch.nn.functional.linear(x, w))
        )

    def test_linear_operation(self):
        x = torch.randn(128, 256, device='cuda')
        w = torch.randn(512, 256, device='cuda')
        b = torch.randn(512, device='cuda')

        self.assertTrue(
            triton.testing.allclose(trident.operation.Linear.apply(x, w, b), torch.nn.functional.linear(x, w, b))
        )

    def test_linear_module_no_bias(self):
        linear0 = torch.nn.Linear(128, 256, bias=False).to('cuda')
        linear1 = trident.module.Linear(128, 256, bias=False).load_state_dict(linear0.state_dict())
        x = torch.randn(128, 128, device='cuda')

        self.assertTrue(triton.testing.allclose(linear0(x), linear1(x)))

    def test_linear_module(self):
        linear0 = torch.nn.Linear(128, 256).to('cuda')
        linear1 = trident.module.Linear(128, 256).load_state_dict(linear0.state_dict())
        x = torch.randn(128, 128, device='cuda')

        self.assertTrue(triton.testing.allclose(linear0(x), linear1(x)))


if __name__ == '__main__':
    unittest.main()
