import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple, Callable, Dict
from gymnasium import spaces
import random

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

class EnvConfig:
    # max_nodes: int = 200
    # top_k: int = 16
    # invalid_action_penalty: float = -1.0
    # no_feasible_penalty: float = -2.0
    # lambda_util: float = 0.1 # weight for utilization improvement
    # # If True, pods in an episode are the order of given list
    # # If False and prob_dist provided, sample from distribution
    # sequential_pods: bool = True
    def __init__(
        self,
        max_nodes: int = 200,
        top_k: Optional[int] = 16,
        invalid_action_penalty: float = -1.0,
        no_feasible_penalty: float = -2.0,
        lambda_util: float = 0.1, # weight for utilization improvement
        sequential_pods: bool = True, # If True, pods in an episode are the order of given list
    ):
        self.max_nodes = max_nodes
        self.top_k = top_k
        self.invalid_action_penalty = invalid_action_penalty
        self.no_feasible_penalty = no_feasible_penalty
        self.lambda_util = lambda_util
        self.sequential_pods = sequential_pods

class K8sScheduleEnv(gym.Env):
    """K8s-like scheduling env with CPU/MEM/GPU resources.
    
    
    Observation:
    dict {
    'nodes': Box(shape=(max_nodes, 3), low=0, high=1), # normalized remaining
    'pod': Box(shape=(3,), low=0, high=1) # normalized w.r.t. a reference capacity
    }
    Action:
    Discrete(max_nodes) # choose a node index
    
    
    Info:
    action_mask: np.ndarray bool [max_nodes]
    candidates: np.ndarray int [<=max_nodes] (Top-K feasible indices)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        node_capacity: np.ndarray, # (N,3) absolute capacity
        pod_list: Optional[List[np.ndarray]] = None, # list of (3,) absolute requests
        pod_prob_dist: Optional[List[Tuple[np.ndarray, float]]] = None, # [(req,(3,), prob)]
        config: EnvConfig = EnvConfig(),
        frag_fn: Callable[[np.ndarray, np.ndarray], float] = default_frag,
    ) -> None:
        super().__init__()
        assert node_capacity.ndim == 2 and node_capacity.shape[1] == 3
        self.N_total = node_capacity.shape[0]
        self.max_nodes = config.max_nodes
        assert self.N_total <= self.max_nodes, "node_capacity exceeds max_nodes"

        # 将节点容量转为 float32 保存
        self.node_capacity_abs = node_capacity.astype(np.float32)
        # 创建一个 (max_nodes,3) 的零数组，用于统一维度
        self.capacity_pad = np.zeros((self.max_nodes, 3), dtype=np.float32)
        # 将实际节点容量复制到零数组的前 N_total 行，其余部分保持为零
        self.capacity_pad[: self.N_total] = self.node_capacity_abs

        # 如果提供了 pod_list，将其中每个 Pod 请求转成 float32 数组；如果未提供，则为空列表
        self.pod_list_abs = [np.asarray(p, dtype=np.float32) for p in (pod_list or [])]
        self.pod_prob_dist = pod_prob_dist # list of (req, prob)
        self.config = config
        self.frag_fn = frag_fn

        # 动作空间是一个离散空间，大小为 max_nodes，表示“选择第几个节点调度 Pod”。
        self.action_space = spaces.Discrete(self.max_nodes)
        # 观测是一个字典，包含两部分：
        # - nodes：每个节点归一化后的剩余资源（max_nodes × 3），值域 [0,1]
        # - pod：当前 Pod 归一化后的资源请求（3，），值域 [0,1]
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes, 3), dtype=np.float32),
                "pod": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

        # stateful
        # 存放每个节点当前的剩余资源量（绝对值）
        self.remaining_abs: np.ndarray = None # (max_nodes,3) absolute remaining + padded zeros
        # 布尔掩码，标识哪些节点是真实存在的
        self.active_nodes_mask: np.ndarray = None # (max_nodes,) bool
        # 存储本轮 episode 中的 Pod 请求序列
        self.curr_pods: List[np.ndarray] = []
        # 当前处理到第几个 Pod 的索引
        self.pod_idx: int = 0
        # 记录上一步的碎片度和资源利用率，用于计算增量奖励（ΔF 和 ΔUtil）。
        self.prev_frag: float = 0.0
        self.prev_util: float = 0.0

    # -------- Episode construction helpers --------
    def _build_episode_pods(self) -> List[np.ndarray]:
        # 如果配置要求“顺序调度” (sequential_pods=True) 且提供了固定 pod_list_abs: 
        # - 直接返回这个列表的副本（避免原数据被修改）。
        if self.config.sequential_pods and self.pod_list_abs:
            return [p.copy() for p in self.pod_list_abs]
        # 如果给了概率分布
        if self.pod_prob_dist:
            reqs, probs = zip(*self.pod_prob_dist)
            reqs = [np.asarray(r, dtype=np.float32) for r in reqs]
            probs = np.asarray(probs, dtype=np.float32)
            probs = probs / probs.sum()
            # Length heuristic: sample ~len(pod_list_abs) if exists else 64
            L = len(self.pod_list_abs) if self.pod_list_abs else 64
            idxs = np.random.choice(len(reqs), size=L, p=probs)
            return [reqs[i].copy() for i in idxs]
        # fallback: small synthetic pods
        L = 32
        reqs = []
        cap_mean = self.node_capacity_abs.mean(axis=0)
        for _ in range(L):
            # 5% ~ 25% of mean capacity per dim
            r = cap_mean * np.random.uniform(0.05, 0.25, size=(3,)).astype(np.float32)
            r[2] = float(np.clip(np.round(r[2]), 0, max(1, int(cap_mean[2])))) # GPU often discrete
            reqs.append(r)
        return reqs

    def _normalize_obs(self) -> Dict[str, np.ndarray]:
        """
        把剩余资源和当前 Pod 需求归一化成 [0,1]，形成环境观测。
        """
        # Normalize remaining by capacity (per-node) to [0,1]; padded nodes are zeros + inactive mask
        eps = 1e-8
        ratio = np.zeros_like(self.remaining_abs, dtype=np.float32)
        mask = self.active_nodes_mask[:, None]
        ratio[mask] = np.clip(
        self.remaining_abs[mask] / (self.capacity_pad[mask] + eps), 0.0, 1.0
        )
        # Normalize pod by global mean capacity to have a stable scale
        cap_mean = self.capacity_pad[self.active_nodes_mask].mean(axis=0)
        pod_norm = np.clip(self.curr_pods[self.pod_idx] / (cap_mean + eps), 0.0, 1.0).astype(
        np.float32
        )
        return {"nodes": ratio, "pod": pod_norm}

    def _feasible_mask(self, pod_req_abs: np.ndarray) -> np.ndarray:
        mask = self.active_nodes_mask.copy()
        # feas：对每个节点检查三维资源是否都 ≥ Pod 需求。
        feas = (self.remaining_abs >= pod_req_abs[None, :]).all(axis=1)
        mask &= feas # 等价于 mask = mask & feas：仅保留既真实又能容纳 Pod 的节点
        return mask # (max_nodes,) bool
    
    def _topk_candidates(self, pod_req_abs: np.ndarray, feas_mask: np.ndarray) -> np.ndarray:
        # Evaluate ΔF for each feasible node by simulating placement
        idxs = np.where(feas_mask)[0]
        if len(idxs) == 0:
            return idxs
        if self.config.top_k is None or self.config.top_k <= 0 or len(idxs) <= self.config.top_k:
            return idxs
        # Compute ΔF
        deltas = []
        for i in idxs:
            new_remaining = self.remaining_abs.copy()
            new_remaining[i, :] -= pod_req_abs
            dF = self.frag_fn(new_remaining[: self.N_total], self.capacity_pad[: self.N_total]) - self.prev_frag
            deltas.append((dF, i))
        deltas.sort(key=lambda x: x[0]) # ascending ΔF
        chosen = [i for _, i in deltas[: self.config.top_k]]
        return np.asarray(chosen, dtype=np.int64)
    
    # -------------- Gym API --------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # 长度为 max_nodes 的布尔掩码，表示哪些节点是“真实存在”的
        self.active_nodes_mask = np.zeros(self.max_nodes, dtype=bool)
        # 前 N_total 个为真（真实节点），其余为假（padding 节点）。
        self.active_nodes_mask[: self.N_total] = True
        # 初始化每个节点的剩余资源矩阵：先全置零，再用节点容量填充真实节点部分。
        self.remaining_abs = np.zeros((self.max_nodes, 3), dtype=np.float32)
        self.remaining_abs[: self.N_total] = self.node_capacity_abs.copy()
        # 构建本轮 episode 的 Pod 请求序列，并从第 0 个开始调度。
        self.curr_pods = self._build_episode_pods()
        self.pod_idx = 0
        # initial frag/util
        self.prev_frag = self.frag_fn(self.remaining_abs[: self.N_total], self.capacity_pad[: self.N_total])
        self.prev_util = compute_utilization(
            self.remaining_abs[: self.N_total], self.capacity_pad[: self.N_total]
        )
        obs = self._normalize_obs()
        pod_req_abs = self.curr_pods[self.pod_idx]
        feas = self._feasible_mask(pod_req_abs)
        cands = self._topk_candidates(pod_req_abs, feas)
        info = {"action_mask": feas, "candidates": cands}
        return obs, info
    
    def step(self, action: int):
        assert isinstance(action, (int, np.integer))
        done = False
        truncated = False
        # 获取当前 Pod 的资源需求
        pod_req_abs = self.curr_pods[self.pod_idx]
        # 生成可行动作掩码 feas
        feas = self._feasible_mask(pod_req_abs)

        if not feas.any():
            # No feasible action -> end episode with penalty
            # 如果没有任何节点能放置 Pod，直接结束 episode，给一个惩罚性奖励
            reward = float(self.config.no_feasible_penalty)
            done = True
            next_obs = self._normalize_obs()
            info = {"action_mask": np.zeros_like(feas), "candidates": np.array([], dtype=np.int64)}
            # 下一观测仍是当前状态，但掩码为空
            return next_obs, reward, done, truncated, info

        if not feas[action]:
            # Invalid action (should be prevented by agent via mask); penalize and continue
            # 如果智能体选了一个不满足资源的节点，给一个较小惩罚并继续。
            reward = float(self.config.invalid_action_penalty)
        else:
            # Apply placement
            # 扣减被选节点的剩余资源
            self.remaining_abs[action, :] -= pod_req_abs
            # Compute reward via ΔF and ΔUtil
            frag = self.frag_fn(self.remaining_abs[: self.N_total], self.capacity_pad[: self.N_total])
            util = compute_utilization(
            self.remaining_abs[: self.N_total], self.capacity_pad[: self.N_total]
            )
            dF = frag - self.prev_frag
            dU = util - self.prev_util
            reward = float(-dF + self.config.lambda_util * dU)
            self.prev_frag, self.prev_util = frag, util
            # Next pod
            self.pod_idx += 1
            if self.pod_idx >= len(self.curr_pods):
                done = True

        next_obs = self._normalize_obs()
        if not done:
            next_req_abs = self.curr_pods[self.pod_idx]
            next_feas = self._feasible_mask(next_req_abs)
            next_cands = self._topk_candidates(next_req_abs, next_feas)
        else:
            next_feas = np.zeros(self.max_nodes, dtype=bool)
            next_cands = np.array([], dtype=np.int64)

        info = {"action_mask": next_feas, "candidates": next_cands}
        return next_obs, reward, done, truncated, info
    
# ============================
# Baselines (for quick reference)
# ============================

def random_baseline(env: K8sScheduleEnv, episodes: int = 10) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            mask = info["action_mask"]
            cands = info["candidates"]
            valid = mask.copy()
            if len(cands) > 0:
                tmp = np.zeros_like(valid)
                tmp[cands] = True
                valid &= tmp
            idxs = np.where(valid)[0]
            if len(idxs) == 0:
                a = 0
            else:
                a = int(np.random.choice(idxs))
            obs, r, done, _, info = env.step(a)
            total += r
    return total / max(1, episodes)


def best_fit_decreasing_baseline(env: K8sScheduleEnv, episodes: int = 10) -> float:
    """Very rough multi-dim BFD: choose node minimizing L2 distance to zero leftover after placement."""
    total = 0.0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            mask = info["action_mask"]
            cands = info["candidates"]
            valid = mask.copy()
            if len(cands) > 0:
                tmp = np.zeros_like(valid)
                tmp[cands] = True
                valid &= tmp
            idxs = np.where(valid)[0]
            if len(idxs) == 0:
                a = 0
            else:
                # pick argmin ||remaining - req||_2
                rem = obs["nodes"][idxs]  # normalized
                req = obs["pod"][None, :]
                d = np.linalg.norm(np.clip(rem - req, 0.0, 1.0), axis=1)
                a = int(idxs[np.argmin(d)])
            obs, r, done, _, info = env.step(a)
            total += r
    return total / max(1, episodes)