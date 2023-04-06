"""
Copyright ⓒ Kakao Brain Corp. All rights reserved.

Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import unittest
import torch
import torch.nn.functional as tf
import triton.testing as tt
import function as tdf


class LinearTestCase(unittest.TestCase):
    def test_small_tensor_no_bias(self):
        x = torch.randn(512, 512, device='cuda')
        w = torch.randn(256, 512, device='cuda')

        self.assertTrue(tt.allclose(tdf.linear(x, w), tf.linear(x, w)))

    def test_large_tensor_no_bias(self):
        x = torch.randn(1024, 2048, device='cuda')
        w = torch.randn(4096, 2048, device='cuda')

        self.assertTrue(tt.allclose(tdf.linear(x, w), tf.linear(x, w)))

    def test_small_tensor(self):
        x = torch.randn(64, 32, device='cuda')
        w = torch.randn(16, 32, device='cuda')
        b = torch.randn(16, device='cuda')

        self.assertTrue(tt.allclose(tdf.linear(x, w, b), tf.linear(x, w, b)))

    def test_large_tensor(self):
        x = torch.randn(2048, 4096, device='cuda')
        w = torch.randn(1024, 4096, device='cuda')
        b = torch.randn(1024, device='cuda')

        self.assertTrue(tt.allclose(tdf.linear(x, w, b), tf.linear(x, w, b)))

    def test_linear_zero_input(self):
        x = torch.full((128, 32), 0.0, device='cuda')
        w = torch.randn(256, 32, device='cuda')
        b = torch.full((1, 256), 0.1, device='cuda')

        self.assertTrue(tt.allclose(tdf.linear(x, w, b), tf.linear(x, w, b)))

    def test_linear_zero_bias(self):
        x = torch.randn(128, 32, device='cuda')
        w = torch.randn(256, 32, device='cuda')
        b = torch.full((1, 256), 0.0, device='cuda')

        self.assertTrue(tt.allclose(tdf.linear(x, w, b), tf.linear(x, w, b)))


if __name__ == '__main__':
    unittest.main()
