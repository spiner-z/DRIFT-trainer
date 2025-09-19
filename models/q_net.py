import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================
# DQN Network & Agent
# ============================
class QNet(nn.Module):
    """Per-node Q scorer conditioned on pod features.

    Inputs:
      nodes: (B, max_nodes, 3) normalized remaining per node
      pod:   (B, 3) normalized request

    Output:
      Q: (B, max_nodes)
    """

    def __init__(self, max_nodes: int, node_dim: int = 3, pod_dim: int = 3, hidden: int = 128):
        super().__init__()
        in_dim = node_dim + pod_dim
        self.max_nodes = max_nodes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, nodes: torch.Tensor, pod: torch.Tensor) -> torch.Tensor:
        B, N, D = nodes.shape
        assert N == self.max_nodes
        pod_expanded = pod.unsqueeze(1).expand(B, N, pod.shape[-1])  # (B,N,3)
        x = torch.cat([nodes, pod_expanded], dim=-1)  # (B,N,6)
        x = x.reshape(B * N, -1)
        q = self.mlp(x).reshape(B, N)
        return q  # (B, N)


class ReplayBuffer:
    def __init__(self, capacity: int, max_nodes: int):
        self.capacity = int(capacity)
        self.max_nodes = max_nodes
        self.reset()

    def reset(self):
        self.mem = []
        self.pos = 0

    def push(
        self,
        obs_nodes: np.ndarray,
        obs_pod: np.ndarray,
        action: int,
        reward: float,
        next_obs_nodes: np.ndarray,
        next_obs_pod: np.ndarray,
        done: bool,
        next_valid_mask: np.ndarray,  # bool [max_nodes]
    ):
        data = (
            obs_nodes.astype(np.float32),
            obs_pod.astype(np.float32),
            int(action),
            float(reward),
            next_obs_nodes.astype(np.float32),
            next_obs_pod.astype(np.float32),
            bool(done),
            next_valid_mask.astype(np.bool_),
        )
        if len(self.mem) < self.capacity:
            self.mem.append(data)
        else:
            self.mem[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.mem, batch_size)
        obs_nodes, obs_pod, acts, rews, next_nodes, next_pod, dones, next_mask = zip(*batch)
        return (
            np.stack(obs_nodes),
            np.stack(obs_pod),
            np.asarray(acts, dtype=np.int64),
            np.asarray(rews, dtype=np.float32),
            np.stack(next_nodes),
            np.stack(next_pod),
            np.asarray(dones, dtype=np.bool_),
            np.stack(next_mask),
        )

    def __len__(self):
        return len(self.mem)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 50000
    train_start: int = 2000
    train_freq: int = 1
    target_sync: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50000
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DQNAgent:
    def __init__(self, max_nodes: int, cfg: DQNConfig):
        self.cfg = cfg
        self.max_nodes = max_nodes
        self.q = QNet(max_nodes).to(cfg.device)
        self.tgt = QNet(max_nodes).to(cfg.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.step_count = 0

    def epsilon(self) -> float:
        s = self.step_count
        e = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * max(
            0.0, 1.0 - s / float(self.cfg.eps_decay_steps)
        )
        return float(e)

    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> int:
        mask: np.ndarray = info.get("action_mask")  # bool [max_nodes]
        cands: np.ndarray = info.get("candidates")  # int  [<=max_nodes]
        valid = mask.copy()
        if cands is not None and len(cands) > 0:
            tmp = np.zeros_like(valid)
            tmp[cands] = True
            valid &= tmp
        valid_idxs = np.where(valid)[0]
        if len(valid_idxs) == 0:
            # fallback: choose any active node (should be handled by env)
            return int(np.random.randint(0, self.max_nodes))

        eps = self.epsilon()
        if random.random() < eps:
            return int(np.random.choice(valid_idxs))
        # Greedy
        self.q.eval()
        nodes = torch.from_numpy(obs["nodes"]).unsqueeze(0).to(self.cfg.device)  # (1,N,3)
        pod = torch.from_numpy(obs["pod"]).unsqueeze(0).to(self.cfg.device)      # (1,3)
        q = self.q(nodes, pod)[0].detach().cpu().numpy()  # (N,)
        # Mask invalid with -inf
        q_masked = q.copy()
        q_masked[~valid] = -1e9
        return int(np.argmax(q_masked))

    def train_step(self, batch):
        (
            obs_nodes,
            obs_pod,
            acts,
            rews,
            next_nodes,
            next_pod,
            dones,
            next_mask,
        ) = batch

        device = self.cfg.device
        obs_nodes_t = torch.from_numpy(obs_nodes).to(device)  # (B,N,3)
        obs_pod_t = torch.from_numpy(obs_pod).to(device)      # (B,3)
        acts_t = torch.from_numpy(acts).to(device).long()     # (B,)
        rews_t = torch.from_numpy(rews).to(device)            # (B,)
        next_nodes_t = torch.from_numpy(next_nodes).to(device)
        next_pod_t = torch.from_numpy(next_pod).to(device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(device)  # (B,)
        next_mask_t = torch.from_numpy(next_mask.astype(np.float32)).to(device)  # (B,N)

        # Current Q(s,a)
        q_all = self.q(obs_nodes_t, obs_pod_t)  # (B,N)
        q_sa = q_all.gather(1, acts_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Target r + gamma * max_a' Q'(s',a') over valid actions only
        with torch.no_grad():
            q_next_all = self.tgt(next_nodes_t, next_pod_t)  # (B,N)
            # set invalid to -inf: use large negative via mask
            invalid = (1.0 - next_mask_t) > 0.5
            q_next_all[invalid] = -1e9
            q_next_max, _ = q_next_all.max(dim=1)  # (B,)
            target = rews_t + self.cfg.gamma * (1.0 - dones_t) * q_next_max

        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return float(loss.item())

    def maybe_sync_target(self):
        if self.step_count % self.cfg.target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())