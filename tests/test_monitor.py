import pytest
import numpy as np

from utils.monitor import default_frag, compute_utilization

def test_default_frag_basic():
    # two nodes, clear leftover
    capacity = np.array([[100.0, 200.0, 2.0],
                         [100.0, 200.0, 2.0]], dtype=np.float32)
    remaining = np.array([[50.0, 100.0, 1.0],
                          [50.0, 100.0, 1.0]], dtype=np.float32)
    f = default_frag(remaining, capacity)
    # ratios each node = [0.5, 0.5, 0.5], std across dims = 0.0, sum should be 0
    assert pytest.approx(f, rel=1e-6) == 0.0

def test_default_frag_padding_and_zero_capacity():
    # second node has zero capacity (should be masked out)
    capacity = np.array([[100.0, 200.0, 2.0],
                         [0.0, 0.0, 0.0]], dtype=np.float32)
    remaining = np.array([[50.0, 100.0, 1.0],
                          [0.0, 0.0, 0.0]], dtype=np.float32)
    f = default_frag(remaining, capacity)
    # only first node contributes, its std across dims is 0
    assert pytest.approx(f, rel=1e-6) == 0.0

def test_compute_utilization_basic():
    capacity = np.array([[10.0, 10.0, 1.0],
                         [10.0, 10.0, 1.0]], dtype=np.float32)
    remaining = np.array([[5.0, 5.0, 1.0],
                          [0.0, 0.0, 0.0]], dtype=np.float32)
    util = compute_utilization(remaining, capacity)
    # used per dim: cpu (15/20)=0.75, mem 0.75, gpu (1/2)=0.5 -> mean = 1/3 * (0.75+0.75+0.5)=2/3
    assert pytest.approx(util, rel=1e-6) == (0.75+0.75+0.5)/3.0