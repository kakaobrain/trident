"""
Copyright ⓒ Kakao Brain Corp. All rights reserved.

Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import triton
import triton.language as tl


@triton.jit
def relu(x):
    return tl.where(x > 0, x, 0)


@triton.jit
def leaky_relu(x, a=0.01):
    return tl.where(x > 0, x, 0) + a * tl.where(x > 0, 0, x)
