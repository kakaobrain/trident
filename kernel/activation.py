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
