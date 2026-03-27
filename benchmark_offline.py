"""
Benchmark offline policy inference on processed UMI data.

Examples:
    python benchmark_offline.py \
        -i cup_wild_vit_l_1img.ckpt \
        -d path/to/dataset.zarr.zip

    python benchmark_offline.py \
        -i cup_wild_vit_l_1img.ckpt \
        -d path/to/session_dir \
        --episode 0 --num-samples 200

This script never sends actions to the robot. It only loads recorded,
SLAM-processed data and measures policy inference latency / throughput.
"""

import os
import pathlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

import click
import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.pose_util import pose_to_mat, mat_to_pose10d

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ReplayBufferBenchmarkDataset:
    """
    Lightweight benchmark dataset for raw replay buffers.

    This mirrors the observation path in `diffusion_policy.dataset.umi_dataset`
    closely enough for inference benchmarking while avoiding assumptions that
    the input must be a zipped training dataset.
    """

    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        pose_repr: Optional[dict] = None,
        action_padding: bool = False,
        max_duration: Optional[float] = None,
    ):
        self.shape_meta = shape_meta
        self.pose_repr = pose_repr or {}
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "relative")

        self.num_robot = 0
        self.rgb_keys = []
        self.lowdim_keys = []
        self.sampler_lowdim_keys = []
        self.key_horizon = {}
        self.key_latency_steps = {}
        self.key_down_sample_steps = {}

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            key_type = attr.get("type", "low_dim")
            if key_type == "rgb":
                self.rgb_keys.append(key)
            elif key_type == "low_dim":
                self.lowdim_keys.append(key)
                if "wrt" not in key:
                    self.sampler_lowdim_keys.append(key)

            if key.endswith("eef_pos"):
                self.num_robot += 1

            self.key_horizon[key] = attr["horizon"]
            self.key_latency_steps[key] = attr["latency_steps"]
            self.key_down_sample_steps[key] = attr["down_sample_steps"]

        self.key_horizon["action"] = shape_meta["action"]["horizon"]
        self.key_latency_steps["action"] = shape_meta["action"]["latency_steps"]
        self.key_down_sample_steps["action"] = shape_meta["action"]["down_sample_steps"]

        for key in replay_buffer.keys():
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split("_")[0] + "_eef_pos"
                self.key_horizon[key] = obs_shape_meta[query_key]["horizon"]
                self.key_latency_steps[key] = obs_shape_meta[query_key]["latency_steps"]
                self.key_down_sample_steps[key] = obs_shape_meta[query_key]["down_sample_steps"]

        self.sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=None,
            action_padding=action_padding,
            repeat_frame_prob=0.0,
            max_duration=max_duration,
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        data = self.sampler.sample_sequence(idx)

        obs_dict = {}
        for key in self.rgb_keys:
            if key not in data:
                continue
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0

        for key in self.sampler_lowdim_keys:
            if key not in data:
                continue
            obs_dict[key] = data[key].astype(np.float32)

        # Generate robot-relative observations when requested by the model.
        for robot_id in range(self.num_robot):
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )

            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue

                pos_key = f"robot{robot_id}_eef_pos_wrt{other_robot_id}"
                rot_key = f"robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}"
                if pos_key not in self.lowdim_keys and rot_key not in self.lowdim_keys:
                    continue

                other_pose_mat = pose_to_mat(
                    np.concatenate(
                        [
                            obs_dict[f"robot{other_robot_id}_eef_pos"],
                            obs_dict[f"robot{other_robot_id}_eef_rot_axis_angle"],
                        ],
                        axis=-1,
                    )
                )
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep="relative",
                    backward=False,
                )
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                if pos_key in self.lowdim_keys:
                    obs_dict[pos_key] = rel_obs_pose[:, :3].astype(np.float32)
                if rot_key in self.lowdim_keys:
                    obs_dict[rot_key] = rel_obs_pose[:, 3:].astype(np.float32)

        # Generate pose relative to the demonstration start when requested.
        for robot_id in range(self.num_robot):
            pos_key = f"robot{robot_id}_eef_pos_wrt_start"
            rot_key = f"robot{robot_id}_eef_rot_axis_angle_wrt_start"
            start_key = f"robot{robot_id}_demo_start_pose"
            if start_key not in obs_dict:
                continue
            if pos_key not in self.lowdim_keys and rot_key not in self.lowdim_keys:
                continue

            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            start_pose_mat = pose_to_mat(obs_dict[start_key][0])
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep="relative",
                backward=False,
            )
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            if pos_key in self.lowdim_keys:
                obs_dict[pos_key] = rel_obs_pose[:, :3].astype(np.float32)
            if rot_key in self.lowdim_keys:
                obs_dict[rot_key] = rel_obs_pose[:, 3:].astype(np.float32)

        # Convert the base robot pose representation exactly as the dataset does.
        for robot_id in range(self.num_robot):
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False,
            )
            obs_pose = mat_to_pose10d(obs_pose_mat)
            obs_dict[f"robot{robot_id}_eef_pos"] = obs_pose[:, :3].astype(np.float32)
            obs_dict[f"robot{robot_id}_eef_rot_axis_angle"] = obs_pose[:, 3:].astype(np.float32)

        # Remove helper keys that are not model inputs.
        for key in list(obs_dict.keys()):
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                del obs_dict[key]

        return {"obs": dict_apply(obs_dict, torch.from_numpy)}


def load_checkpoint(ckpt_path: str) -> Tuple[dict, BaseWorkspace, torch.nn.Module]:
    ckpt_path = os.path.expanduser(ckpt_path)
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")

    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    return cfg, workspace, policy


def resolve_dataset_path(data_path: str) -> pathlib.Path:
    path = pathlib.Path(os.path.expanduser(data_path))
    if path.is_file():
        return path

    if path.is_dir():
        candidates = [
            path / "dataset.zarr.zip",
            path / "replay_buffer.zarr",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Could not find a dataset at {path}. Expected either a direct "
        f"`dataset.zarr.zip` / `replay_buffer.zarr` path or a directory containing one."
    )


def create_dataset(cfg, dataset_path: pathlib.Path):
    dataset_path_str = str(dataset_path)
    suffixes = "".join(dataset_path.suffixes)

    if suffixes.endswith(".zarr.zip"):
        return hydra.utils.instantiate(
            cfg.task.dataset,
            dataset_path=dataset_path_str,
            cache_dir=None,
            val_ratio=0.0,
            repeat_frame_prob=0.0,
        )

    replay_buffer = ReplayBuffer.create_from_path(dataset_path_str, mode="r")
    return ReplayBufferBenchmarkDataset(
        shape_meta=cfg.task.shape_meta,
        replay_buffer=replay_buffer,
        pose_repr=cfg.task.get("pose_repr", {}),
        action_padding=cfg.task.dataset.get("action_padding", False),
        max_duration=cfg.task.dataset.get("max_duration", None),
    )


def get_episode_ids(indices: Sequence[Tuple[int, int, int, bool]]) -> List[int]:
    episode_ids = []
    episode_lookup = {}
    next_episode_id = 0
    for _, start_idx, end_idx, _ in indices:
        key = (start_idx, end_idx)
        if key not in episode_lookup:
            episode_lookup[key] = next_episode_id
            next_episode_id += 1
        episode_ids.append(episode_lookup[key])
    return episode_ids


def select_sample_indices(
    dataset,
    episode: Optional[int],
    start_index: int,
    num_samples: Optional[int],
    stride: int,
) -> Tuple[List[int], int]:
    if not hasattr(dataset, "sampler") or not hasattr(dataset.sampler, "indices"):
        indices = list(range(len(dataset)))
        indices = indices[start_index::stride]
        if num_samples is not None:
            indices = indices[:num_samples]
        return indices, -1

    episode_ids = get_episode_ids(dataset.sampler.indices)
    total_episodes = 0 if not episode_ids else max(episode_ids) + 1

    candidate_indices = list(range(len(dataset)))
    if episode is not None:
        candidate_indices = [i for i in candidate_indices if episode_ids[i] == episode]

    candidate_indices = candidate_indices[start_index::stride]
    if num_samples is not None:
        candidate_indices = candidate_indices[:num_samples]

    return candidate_indices, total_episodes


def move_obs_to_device(obs_cpu: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return dict_apply(obs_cpu, lambda x: x.unsqueeze(0).to(device, non_blocking=True))


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_policy(
    policy,
    samples_cpu: List[Dict[str, torch.Tensor]],
    samples_device: Optional[List[Dict[str, torch.Tensor]]],
    device: torch.device,
    warmup: int,
) -> Dict[str, List[float]]:
    results = {
        "transfer_and_predict_ms": [],
        "predict_only_ms": [],
    }

    warmup = min(warmup, len(samples_cpu))
    with torch.inference_mode():
        policy.reset()
        for i in range(warmup):
            obs_device = move_obs_to_device(samples_cpu[i], device)
            _ = policy.predict_action(obs_device)
        maybe_sync(device)

        policy.reset()
        for obs_cpu in samples_cpu:
            maybe_sync(device)
            start_t = time.perf_counter()
            obs_device = move_obs_to_device(obs_cpu, device)
            _ = policy.predict_action(obs_device)
            maybe_sync(device)
            end_t = time.perf_counter()
            results["transfer_and_predict_ms"].append((end_t - start_t) * 1000.0)

        if samples_device is not None:
            policy.reset()
            for obs_device in samples_device:
                maybe_sync(device)
                start_t = time.perf_counter()
                _ = policy.predict_action(obs_device)
                maybe_sync(device)
                end_t = time.perf_counter()
                results["predict_only_ms"].append((end_t - start_t) * 1000.0)

    return results


def summarize_latency(name: str, values_ms: List[float]) -> Optional[str]:
    if not values_ms:
        return None

    values = np.asarray(values_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    p50_ms = float(np.percentile(values, 50))
    p95_ms = float(np.percentile(values, 95))
    p99_ms = float(np.percentile(values, 99))
    min_ms = float(np.min(values))
    max_ms = float(np.max(values))
    hz = float(1000.0 / mean_ms)

    return (
        f"{name}\n"
        f"  mean: {mean_ms:.2f} ms ({hz:.2f} Hz)\n"
        f"  p50 : {p50_ms:.2f} ms\n"
        f"  p95 : {p95_ms:.2f} ms\n"
        f"  p99 : {p99_ms:.2f} ms\n"
        f"  min : {min_ms:.2f} ms\n"
        f"  max : {max_ms:.2f} ms"
    )


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option(
    "--data",
    "-d",
    required=True,
    help="Path to dataset.zarr.zip, replay_buffer.zarr, or a directory containing one",
)
@click.option("--episode", type=int, default=None, help="Benchmark a single episode only")
@click.option("--start-index", default=0, type=int, help="Start from this sequence index")
@click.option("--num-samples", "-n", default=200, type=int, help="Number of sequence windows to benchmark")
@click.option("--stride", default=1, type=int, help="Subsample sequence windows by this stride")
@click.option("--warmup", default=20, type=int, help="Number of warmup iterations")
@click.option("--device", default=None, type=str, help="Torch device, e.g. cuda:0 or cpu")
@click.option(
    "--num-inference-steps",
    type=int,
    default=None,
    help="Override diffusion/DDIM inference steps from the checkpoint",
)
def main(input, data, episode, start_index, num_samples, stride, warmup, device, num_inference_steps):
    torch.backends.cudnn.benchmark = True

    cfg, _, policy = load_checkpoint(input)
    dataset_path = resolve_dataset_path(data)
    dataset = create_dataset(cfg, dataset_path)

    if len(dataset) == 0:
        raise RuntimeError("The selected dataset produced zero benchmarkable samples.")

    sample_indices, total_episodes = select_sample_indices(
        dataset=dataset,
        episode=episode,
        start_index=start_index,
        num_samples=num_samples,
        stride=stride,
    )
    if not sample_indices:
        raise RuntimeError("No samples matched the requested episode / slicing options.")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = num_inference_steps

    policy.eval().to(device)

    print(f"Checkpoint: {os.path.expanduser(input)}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"Samples selected: {len(sample_indices)} / {len(dataset)}")
    if total_episodes >= 0:
        print(f"Episodes available: {total_episodes}")
    if episode is not None:
        print(f"Episode filter: {episode}")
    if hasattr(policy, "num_inference_steps"):
        print(f"num_inference_steps: {policy.num_inference_steps}")

    samples_cpu = [dataset[idx]["obs"] for idx in sample_indices]
    first_obs = samples_cpu[0]
    print("Observation keys:")
    for key, value in first_obs.items():
        print(f"  {key}: {tuple(value.shape)}")

    samples_device = None
    if device.type == "cuda":
        try:
            samples_device = [move_obs_to_device(obs, device) for obs in samples_cpu]
            maybe_sync(device)
        except RuntimeError as exc:
            print(f"Skipping predict-only benchmark due to device memory issue: {exc}")
            samples_device = None

    results = benchmark_policy(
        policy=policy,
        samples_cpu=samples_cpu,
        samples_device=samples_device,
        device=device,
        warmup=warmup,
    )

    print("")
    summary = summarize_latency("Transfer + predict", results["transfer_and_predict_ms"])
    if summary is not None:
        print(summary)

    summary = summarize_latency("Predict only", results["predict_only_ms"])
    if summary is not None:
        print("")
        print(summary)


if __name__ == "__main__":
    main()
