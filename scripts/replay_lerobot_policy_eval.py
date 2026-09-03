#!/usr/bin/env python3
"""Replay a LeRobot policy on recorded LeRobot data and report action errors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401 - registers "act"
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401 - registers "pi0"


def parse_episode_list(value: str) -> list[int]:
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return episodes


def prepare_batch(item: dict, camera_keys: list[str]) -> dict[str, Any]:
    batch = {key: value.unsqueeze(0) for key, value in item.items() if torch.is_tensor(value)}
    if isinstance(item.get("task"), str):
        batch["task"] = item["task"]
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].float() / 255.0
    return batch


def make_processors(cfg, dataset_stats: dict):
    if cfg.type == "act":
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        return make_act_pre_post_processors(cfg, dataset_stats=dataset_stats)
    if cfg.type == "pi0":
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        return make_pi0_pre_post_processors(cfg, dataset_stats=dataset_stats)
    raise ValueError(f"unsupported policy type for replay evaluation: {cfg.type}")


def make_relative_no_motion_action(item: dict, target: torch.Tensor) -> torch.Tensor:
    """Build xyz=0, rotation=identity, gripper=persistent action blocks."""
    if target.numel() % 10 != 0:
        raise ValueError(
            f"relative no-motion baseline requires 10D action blocks, got {target.numel()} values"
        )
    state = item["observation.state"].detach().cpu().flatten()
    if state.numel() % 10 != 0:
        raise ValueError(
            f"relative no-motion baseline requires 10D state blocks, got {state.numel()} values"
        )

    baseline = torch.zeros_like(target)
    num_state_arms = state.numel() // 10
    for block_idx in range(target.numel() // 10):
        offset = block_idx * 10
        arm_idx = block_idx % num_state_arms
        baseline[offset + 3] = 1.0
        baseline[offset + 7] = 1.0
        baseline[offset + 9] = state[arm_idx * 10 + 9]
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="data")
    parser.add_argument("--episodes", default="0-4", help="Examples: 0, 0-4, 0,3,8")
    parser.add_argument("--max-steps", default=50, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-action-steps", default=1, type=int)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--mode", choices=["loss", "select_action"], default="select_action")
    parser.add_argument("--loss-batches", default=20, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    cfg.pretrained_path = args.policy_path
    cfg.n_action_steps = args.policy_action_steps
    if hasattr(cfg, "pretrained_backbone_weights"):
        # The checkpoint already has trained backbone weights. Avoid torchvision downloads.
        cfg.pretrained_backbone_weights = None

    if args.mode == "loss":
        from torch.utils.data import DataLoader

        meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
        delta_timestamps = resolve_delta_timestamps(cfg, meta)
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )
        policy = make_policy(cfg, ds_meta=dataset.meta)
        policy.eval()
        preprocessor, _ = make_processors(cfg, dataset.meta.stats)

        losses: list[float] = []
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= args.loss_batches:
                    break
                for cam_key in dataset.meta.camera_keys:
                    if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                        batch[cam_key] = batch[cam_key].float() / 255.0
                batch = preprocessor(batch)
                loss, out = policy.forward(batch)
                losses.append(float(loss))
                print(f"batch {batch_idx:02d}: loss={float(loss):.5f} l1={float(out['l1_loss']):.5f}")

        print("---")
        print(f"batches: {len(losses)}")
        print(f"mean_loss: {sum(losses) / len(losses):.5f}")
        print(f"min_loss: {min(losses):.5f}")
        print(f"max_loss: {max(losses):.5f}")
        return

    episodes = parse_episode_list(args.episodes)
    meta_dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )
    policy = make_policy(cfg, ds_meta=meta_dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_processors(cfg, meta_dataset.meta.stats)

    all_errs: list[torch.Tensor] = []
    all_diffs: list[torch.Tensor] = []
    all_baseline_errs: list[torch.Tensor] = []
    for ep in episodes:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            episodes=[ep],
            return_uint8=True,
            video_backend=args.video_backend,
        )
        policy.reset()
        ep_errs: list[torch.Tensor] = []
        ep_diffs: list[torch.Tensor] = []
        ep_baseline_errs: list[torch.Tensor] = []
        steps = min(args.max_steps, len(dataset))
        with torch.no_grad():
            for step in range(steps):
                item = dataset[step]
                batch = prepare_batch(item, dataset.meta.camera_keys)
                obs = preprocessor(batch)
                pred = policy.select_action(obs)
                pred_raw = postprocessor(pred).squeeze(0).cpu()
                target_raw = item["action"].cpu()
                diff = pred_raw - target_raw
                ep_diffs.append(diff)
                ep_errs.append(diff.abs())
                no_motion = make_relative_no_motion_action(item, target_raw)
                ep_baseline_errs.append((no_motion - target_raw).abs())
        ep_err = torch.stack(ep_errs)
        ep_diff = torch.stack(ep_diffs)
        ep_baseline = torch.stack(ep_baseline_errs)
        all_errs.append(ep_err)
        all_diffs.append(ep_diff)
        all_baseline_errs.append(ep_baseline)
        print(
            f"episode{ep}: steps={steps} "
            f"mean_abs={ep_err.mean().item():.5f} "
            f"no_motion_baseline_abs={ep_baseline.mean().item():.5f} "
            f"median_abs={ep_err.median().item():.5f} "
            f"max_abs={ep_err.max().item():.5f}"
        )

    all_err = torch.cat(all_errs, dim=0)
    all_diff = torch.cat(all_diffs, dim=0)
    all_baseline = torch.cat(all_baseline_errs, dim=0)
    baseline_mean = all_baseline.mean().item()
    print("---")
    print(f"total_steps: {all_err.shape[0]}")
    print(f"mean_abs_all: {all_err.mean().item():.5f}")
    print(f"no_motion_baseline_abs_all: {baseline_mean:.5f}")
    if baseline_mean > 0.0:
        print(
            "relative_improvement_vs_no_motion: "
            f"{1.0 - all_err.mean().item() / baseline_mean:.4f}"
        )
    else:
        print("relative_improvement_vs_no_motion: undefined (baseline error is zero)")
    print(f"median_abs_all: {all_err.median().item():.5f}")
    print(f"max_abs_all: {all_err.max().item():.5f}")
    print(f"mean_signed_per_dim: {all_diff.mean(dim=0).tolist()}")
    print(f"mean_abs_per_dim: {all_err.mean(dim=0).tolist()}")
    print(f"no_motion_baseline_abs_per_dim: {all_baseline.mean(dim=0).tolist()}")


if __name__ == "__main__":
    main()
