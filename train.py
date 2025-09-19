import csv
import numpy as np
import random
from typing import List
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

from gym_envs.k8s_schedule_env import K8sScheduleEnv, EnvConfig, default_frag, random_baseline, best_fit_decreasing_baseline
from models.q_net import QNet, ReplayBuffer, DQNAgent, DQNConfig


# ============================
# Utilities: loading CSVs or make synthetic
# ============================
def load_nodes_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append([float(row["cpu"]), float(row["mem"]), float(row["gpu"])])
    arr = np.asarray(rows, dtype=np.float32)
    return arr

def load_pods_csv(path: str) -> List[np.ndarray]:
    rows = []
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(np.asarray([float(row["cpu_req"]), float(row["mem_req"]), float(row["gpu_req"])], dtype=np.float32))
    return rows

def make_synthetic_nodes(n: int = 20) -> np.ndarray:
    cap = []
    for _ in range(n):
        cpu = np.random.randint(16, 65) # 16 ~ 64
        mem = np.random.randint(64, 257) # 64 ~ 256
        gpu = np.random.randint(0, 8) # 0 ~ 8 GPUs
        cap.append([cpu, mem, gpu])
    return np.asarray(cap, dtype=np.float32)

# ============================
# Training / Evaluation Script
# ============================

def train(env: K8sScheduleEnv, episodes: int, agent: DQNAgent, buffer: ReplayBuffer, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    cfg = agent.cfg
    returns = []
    losses = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            a = agent.select_action(obs, info)
            next_obs, r, done, _, next_info = env.step(a)
            # Build next_valid_mask = action_mask ∧ (optional) candidates
            mask = next_info["action_mask"].copy()
            cands = next_info["candidates"]
            if cands is not None and len(cands) > 0:
                tmp = np.zeros_like(mask)
                tmp[cands] = True
                mask &= tmp

            buffer.push(
                obs["nodes"], obs["pod"], a, r,
                next_obs["nodes"], next_obs["pod"], done,
                mask,
            )

            obs, info = next_obs, next_info
            ep_ret += r

            # Train
            agent.step_count += 1
            if len(buffer) >= cfg.train_start and agent.step_count % cfg.train_freq == 0:
                batch = buffer.sample(cfg.batch_size)
                loss = agent.train_step(batch)
                losses.append(loss)
            agent.maybe_sync_target()

        returns.append(ep_ret)
        if ep % max(1, episodes // 20) == 0:
            avg_ret = sum(returns[-10:]) / max(1, len(returns[-10:]))
            avg_loss = sum(losses[-100:]) / max(1, len(losses[-100:])) if losses else 0.0
            print(f"[Ep {ep:4d}] avg_return(10)={avg_ret:.4f}  avg_loss(100)={avg_loss:.6f}  eps={agent.epsilon():.3f}")

        # checkpoint occasionally
        if ep % max(10, episodes // 10) == 0:
            ckpt_path = os.path.join(outdir, "ckpt.pt")
            torch.save({"q": agent.q.state_dict(), "tgt": agent.tgt.state_dict()}, ckpt_path)

    # final save
    ckpt_path = os.path.join(outdir, "ckpt.pt")
    torch.save({"q": agent.q.state_dict(), "tgt": agent.tgt.state_dict()}, ckpt_path)
    print(f"Saved final checkpoint to: {ckpt_path}")
    return returns, losses


def evaluate(env: K8sScheduleEnv, agent: DQNAgent, episodes: int = 50):
    cfg = agent.cfg
    old_eps_start, old_eps_end = cfg.eps_start, cfg.eps_end
    cfg.eps_start = cfg.eps_end = 0.0
    total = 0.0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            a = agent.select_action(obs, info)
            obs, r, done, _, info = env.step(a)
            total += r
    cfg.eps_start, cfg.eps_end = old_eps_start, old_eps_end
    return total / max(1, episodes)


# ============================
# Main CLI
# ============================

def build_env_from_args(args) -> K8sScheduleEnv:
    if args.nodes_csv and os.path.exists(args.nodes_csv):
        node_cap = load_nodes_csv(args.nodes_csv)
    else:
        node_cap = make_synthetic_nodes(n=min(args.max_nodes, 20))

    pod_list = None
    pod_prob = None
    if args.pods_csv and os.path.exists(args.pods_csv):
        print("Using pods CSV for fixed pod sequence per episode.")
        pod_list = load_pods_csv(args.pods_csv)
    elif args.pod_dist_json and os.path.exists(args.pod_dist_json):
        print("Using pod distribution JSON for random pod sampling per episode.")
        with open(args.pod_dist_json, "r") as f:
            data = json.load(f)
        # expect: [{"req": [cpu,mem,gpu], "prob": p}, ...]
        pod_prob = [(np.asarray(x["req"], dtype=np.float32), float(x["prob"])) for x in data]

    cfg = EnvConfig(
        max_nodes=args.max_nodes,
        top_k=args.top_k,
        invalid_action_penalty=args.invalid_action_penalty,
        no_feasible_penalty=args.no_feasible_penalty,
        lambda_util=args.lambda_util,
        sequential_pods=args.sequential_pods,
    )

    # Let users swap in their own fragmentation function via a separate module path
    frag_fn = default_frag
    print("=== Environment Configuration ===")
    print(f"{len(node_cap)} nodes")
    if pod_list is not None:
        print(f"{len(pod_list)} pods (fixed sequence)")
    elif pod_prob is not None:
        print(f"{len(pod_prob)} pod types (random sampling)")
    else:
        print("Synthetic pods (random sampling)")

    return K8sScheduleEnv(node_cap, pod_list, pod_prob, cfg, frag_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes_csv", type=str, default=None)
    parser.add_argument("--pods_csv", type=str, default=None)
    parser.add_argument("--pod_dist_json", type=str, default=None, help="Distribution JSON: [{req:[c,m,g], prob:0.2}, ...]")
    parser.add_argument("--max_nodes", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--lambda_util", type=float, default=0.1)
    parser.add_argument("--invalid_action_penalty", type=float, default=-1.0)
    parser.add_argument("--no_feasible_penalty", type=float, default=-2.0)
    parser.add_argument("--sequential_pods", action="store_true", help="Use pods CSV order per episode (if provided)")

    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="runs")

    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_start", type=int, default=2000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--target_sync", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=50000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    env = build_env_from_args(args)
    dqn_cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_start=args.train_start,
        train_freq=args.train_freq,
        target_sync=args.target_sync,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        grad_clip=args.grad_clip,
    )

    agent = DQNAgent(env.max_nodes, dqn_cfg)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=agent.cfg.device)
        agent.q.load_state_dict(ckpt["q"])  # type: ignore[index]
        agent.tgt.load_state_dict(ckpt["tgt"])  # type: ignore[index]
        print(f"Loaded checkpoint from {args.checkpoint}")

    if args.evaluate:
        ret = evaluate(env, agent, episodes=50)
        print(f"Evaluation average return: {ret:.4f}")
        return

    buffer = ReplayBuffer(capacity=args.buffer_size, max_nodes=env.max_nodes)

    # quick baselines (optional):
    try:
        rb = random_baseline(env, episodes=5)
        bfd = best_fit_decreasing_baseline(env, episodes=5)
        print(f"Random baseline avg return: {rb:.4f}")
        print(f"BFD    baseline avg return: {bfd:.4f}")
    except Exception as e:
        print(f"Baseline eval skipped due to error: {e}")

    returns, losses = train(env, episodes=args.episodes, agent=agent, buffer=buffer, outdir=args.outdir)
    # print(f"Training returns: {returns}")
    # print(f"Training losses: {losses}")
    print("Training done.")


if __name__ == "__main__":
    main()
