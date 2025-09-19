import numpy as np

def default_frag(remaining: np.ndarray, capacity: np.ndarray) -> float:
    """A simple, replaceable fragmentation metric.


    remaining/capacity -> leftover ratios in [0,1]. For each node, compute
    std across dims (CPU/MEM/GPU). Sum across nodes -> higher means more
    uneven leftover (i.e., more fragmentation).


    Parameters
    ----------
    remaining: (N, 3) remaining absolute amounts per node
    capacity: (N, 3) capacity absolute amounts per node


    Returns
    -------
    float: fragmentation score (lower is better)
    """
    eps = 1e-8
    mask = (capacity.sum(axis=1) > eps)
    if mask.sum() == 0:
        return 0.0
    rem = remaining[mask]
    cap = capacity[mask]
    ratios = np.clip(rem / (cap + eps), 0.0, 1.0) # (M, 3)
    node_std = ratios.std(axis=1) # (M,)
    return float(node_std.sum())

def compute_utilization(remaining: np.ndarray, capacity: np.ndarray) -> float:
    """Average utilization across three dims (CPU/MEM/GPU), aggregated over nodes.
    Returns value in [0,1]."""
    eps = 1e-8
    used = np.clip(capacity - remaining, 0.0, None)
    cap_sum = capacity.sum(axis=0) + eps
    util_by_dim = (used.sum(axis=0) / cap_sum) # 3 dims
    return float(util_by_dim.mean())