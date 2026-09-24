"""公共工具：在没有 GPU 的环境里自动启用 numba 的 CUDA 模拟器。"""
import os

# 必须在 import numba.cuda 之前设置
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402


def check(name, got, expected, **kw):
    ok = np.allclose(got, expected, **kw)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        diff = np.argwhere(~np.isclose(got, expected, **kw))
        print(f"  first mismatch at {diff[0].tolist()}: got {got[tuple(diff[0])]}, expected {expected[tuple(diff[0])]}")
    return ok
