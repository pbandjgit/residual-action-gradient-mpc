"""Dataset utilities for CSTR surrogate training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceFormat:
    history_len: int = 20
    horizon: int = 10
    stride: int = 1
    state_columns: Tuple[str, ...] = ("C_A", "T")
    input_columns: Tuple[str, ...] = ("F", "T_c")
    normalize: bool = True


def load_sequence_format(path: Path) -> SequenceFormat:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return SequenceFormat(
        history_len=cfg["history_len"],
        horizon=cfg["horizon"],
        stride=cfg["stride"],
        state_columns=tuple(cfg["state_columns"]),
        input_columns=tuple(cfg["input_columns"]),
        normalize=cfg["normalize"],
    )


def compute_normalization_stats(
    df: pd.DataFrame,
    state_columns: Tuple[str, ...],
    input_columns: Tuple[str, ...],
) -> Dict:
    state_mean = df[list(state_columns)].mean().to_numpy(dtype=np.float32)
    state_std  = df[list(state_columns)].std().replace(0, 1.0).to_numpy(dtype=np.float32)
    input_mean = df[list(input_columns)].mean().to_numpy(dtype=np.float32)
    input_std  = df[list(input_columns)].std().replace(0, 1.0).to_numpy(dtype=np.float32)
    return {
        "state": {"mean": state_mean.tolist(), "std": state_std.tolist()},
        "input": {"mean": input_mean.tolist(), "std": input_std.tolist()},
    }


def save_normalization_stats(stats: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


class CSTRSequenceDataset(Dataset):
    """Episode-aware sequence dataset for CSTR surrogate training."""

    def __init__(
        self,
        csv_path: Path,
        seq_format: SequenceFormat,
        normalization_stats: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        sample_seed: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.seq_format = seq_format

        df = pd.read_csv(self.csv_path)
        self.state_columns = list(seq_format.state_columns)
        self.input_columns = list(seq_format.input_columns)

        self.state_values   = df[self.state_columns].to_numpy(dtype=np.float32)
        self.input_values   = df[self.input_columns].to_numpy(dtype=np.float32)
        self.episode_values = df["episode"].to_numpy()

        if seq_format.normalize:
            if normalization_stats is None:
                normalization_stats = compute_normalization_stats(
                    df, seq_format.state_columns, seq_format.input_columns
                )
            self.normalization_stats = normalization_stats
            self.state_mean = np.array(normalization_stats["state"]["mean"], dtype=np.float32)
            self.state_std  = np.array(normalization_stats["state"]["std"],  dtype=np.float32)
            self.input_mean = np.array(normalization_stats["input"]["mean"], dtype=np.float32)
            self.input_std  = np.array(normalization_stats["input"]["std"],  dtype=np.float32)
        else:
            self.normalization_stats = None
            self.state_mean = np.zeros(len(self.state_columns), dtype=np.float32)
            self.state_std  = np.ones(len(self.state_columns),  dtype=np.float32)
            self.input_mean = np.zeros(len(self.input_columns), dtype=np.float32)
            self.input_std  = np.ones(len(self.input_columns),  dtype=np.float32)

        self.samples: List[Tuple[int, int]] = []
        self._build_index()

        if max_samples is not None and len(self.samples) > max_samples:
            rng = np.random.default_rng(sample_seed)
            idx = rng.choice(len(self.samples), size=max_samples, replace=False)
            idx.sort()
            self.samples = [self.samples[i] for i in idx]

    def _build_index(self) -> None:
        hist    = self.seq_format.history_len
        horizon = self.seq_format.horizon
        stride  = self.seq_format.stride

        for episode in np.unique(self.episode_values):
            idx = np.where(self.episode_values == episode)[0]
            n = len(idx)
            max_start = n - hist - horizon + 1
            if max_start <= 0:
                continue
            for local_start in range(0, max_start, stride):
                self.samples.append((int(idx[0]), local_start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        base_idx, local_start = self.samples[item]
        hist    = self.seq_format.history_len
        horizon = self.seq_format.horizon

        hs = slice(base_idx + local_start,          base_idx + local_start + hist)
        fs = slice(base_idx + local_start + hist,   base_idx + local_start + hist + horizon)

        state_history = (self.state_values[hs] - self.state_mean) / self.state_std
        input_history = (self.input_values[hs] - self.input_mean) / self.input_std
        future_input  = (self.input_values[fs] - self.input_mean) / self.input_std
        future_state  = (self.state_values[fs] - self.state_mean) / self.state_std

        return {
            "state_history":       torch.from_numpy(state_history),
            "input_history":       torch.from_numpy(input_history),
            "future_input":        torch.from_numpy(future_input),
            "target_future_state": torch.from_numpy(future_state),
        }
