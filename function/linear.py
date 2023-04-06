"""
Copyright ⓒ Kakao Brain Corp. All rights reserved.

Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import torch
import triton
import kernel as tdk


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor=None):
    """
    Applies a linear transformation on the input tensor x using the weight tensor w
    and the bias tensor b, and returns the result.

    :param x: Input tensor. The tensor shape is (*, in_features).
    :param w: Weight tensor. The tensor shape is (out_features, in_features).
    :param b: Bias tensor. The tensor shape is (out_features).
    :return: Output tensor. The tensor shape is (*,out_features).
    """
    assert x.is_cuda and x.is_contiguous()
    assert w.is_cuda and w.is_contiguous()
    assert x.shape[1] == w.shape[1]

    if b is not None:
        assert b.is_cuda and b.is_contiguous()
        assert w.shape[0] == b.shape[0] if b.dim() == 1 else b.shape[1]

    m, k = x.shape
    n, _ = w.shape
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']), triton.cdiv(n, meta['BLOCK_SIZE_N']), )
    y = torch.empty((m, n), device='cuda')

    if b is None:
        tdk.linear[grid](x, x.stride(0), x.stride(1),
                         y, y.stride(0), y.stride(1),
                         w, w.stride(0), w.stride(1),
                         None, 0,
                         m, k, n,
                         BLOCK_SIZE_K=16)
    else:
        assert b.is_cuda and b.is_contiguous()
        assert w.shape[0] == b.shape[0] if b.dim() == 1 else b.shape[1]

        tdk.linear[grid](x, x.stride(0), x.stride(1),
                         y, y.stride(0), y.stride(1),
                         w, w.stride(0), w.stride(1),
                         b, b.stride(0) if b.dim() == 1 else b.stride(1),
                         m, k, n,
                         BLOCK_SIZE_K=16)

    return y
