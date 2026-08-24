#!/usr/bin/env python3
"""Replay a LeRobot policy on recorded LeRobot data and report action errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors


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


def prepare_batch(item: dict, camera_keys: list[str]) -> dict[str, torch.Tensor]:
    batch = {key: value.unsqueeze(0) for key, value in item.items() if torch.is_tensor(value)}
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].float() / 255.0
    return batch


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
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=args.policy_path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": args.device}},
        )

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
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )

    all_errs: list[torch.Tensor] = []
    all_diffs: list[torch.Tensor] = []
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
        ep_err = torch.stack(ep_errs)
        ep_diff = torch.stack(ep_diffs)
        all_errs.append(ep_err)
        all_diffs.append(ep_diff)
        print(
            f"episode{ep}: steps={steps} "
            f"mean_abs={ep_err.mean().item():.5f} "
            f"median_abs={ep_err.median().item():.5f} "
            f"max_abs={ep_err.max().item():.5f}"
        )

    all_err = torch.cat(all_errs, dim=0)
    all_diff = torch.cat(all_diffs, dim=0)
    print("---")
    print(f"total_steps: {all_err.shape[0]}")
    print(f"mean_abs_all: {all_err.mean().item():.5f}")
    print(f"median_abs_all: {all_err.median().item():.5f}")
    print(f"max_abs_all: {all_err.max().item():.5f}")
    print(f"mean_signed_per_dim: {all_diff.mean(dim=0).tolist()}")
    print(f"mean_abs_per_dim: {all_err.mean(dim=0).tolist()}")


if __name__ == "__main__":
    main()
