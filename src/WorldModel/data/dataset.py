"""
World Model Dataset
===================
PyTorch dataset for loading training samples collected by WorldModelDataCollector.

V5 additions:
  - split_by_group(): candidate_group_id-level train/val/test split
  - build_pairwise_data(): construct ranking pairs from samples
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from WorldModel.costs import compute_realized_cost


def stable_candidate_group_key(sample: dict):
    """Return a collision-free group key across fused simulation runs."""
    group_id = str(sample.get("candidate_group_id", ""))
    run_id = str(sample.get("run_id", ""))
    simulation_seed = sample.get("simulation_seed")
    if run_id or simulation_seed is not None:
        seed_key = "" if simulation_seed is None else str(simulation_seed)
        return (run_id, seed_key, group_id)
    return ("legacy_without_provenance", "", group_id)


class WorldModelDataset(Dataset):
    """Dataset wrapping pre-collected world model training samples.

    Each sample is a dict with tensors:
      - node_history:         (L, N, 10)
      - edge_index:           (2, E)
      - edge_features:        (E, 6)
      - demand_context:       (demand_dim,)
      - action_node:          (N, 8)
      - action_global:        (6,)
      - action_edge:          (E, 4)
      - future_node_labels:   (K, N, 6)
      - future_system_labels: (K, 7)
    """

    def __init__(self, samples: List[dict]):
        self.samples = samples

    @classmethod
    def from_file(cls, path: str,
                  long_risk_path: Optional[str] = None) -> "WorldModelDataset":
        """Load dataset, optionally merging long-risk labels.

        Parameters
        ----------
        path : str
            Path to main training data (.pt file).
        long_risk_path : str or None
            Path to long_risk_labels.pt. Labels are merged into samples
            by candidate_key. Samples without matching long-risk labels
            will not have 'long_risk_labels' and the long-risk loss will
            be skipped for them.
        """
        samples = torch.load(path, weights_only=False)
        ds = cls(samples)
        if long_risk_path is not None:
            ds.merge_long_risk_labels(long_risk_path)
        return ds

    def merge_long_risk_labels(self, long_risk_path: str):
        """Merge long-risk labels into samples by candidate_key."""
        lr_data = torch.load(long_risk_path, weights_only=False)
        lr_map = {}
        for entry in lr_data:
            key = entry.get("candidate_key")
            if key:
                lr_map[key] = entry

        matched = 0
        for sample in self.samples:
            key = sample.get("candidate_key")
            if key and key in lr_map:
                lr = lr_map[key]
                sample["long_risk_labels"] = {
                    "risk_peak": torch.tensor(lr["risk_peak"], dtype=torch.float32),
                    "risk_cvar": torch.tensor(lr["risk_cvar"], dtype=torch.float32),
                    "risk_terminal": torch.tensor(lr["risk_terminal"], dtype=torch.float32),
                    "risk_delta_group": torch.tensor(lr["risk_delta_group"], dtype=torch.float32),
                    "risk_event": torch.tensor(lr["risk_event"], dtype=torch.float32),
                }
                matched += 1

        total = len(self.samples)
        print(f"  Long-risk merge: {matched}/{total} samples matched "
              f"({matched/max(total,1)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _group_samples(self) -> Dict[tuple, List[dict]]:
        """Group samples by run/seed-qualified candidate context."""
        groups: Dict[tuple, List[dict]] = {}
        for s in self.samples:
            gid = stable_candidate_group_key(s)
            groups.setdefault(gid, []).append(s)
        return groups

    def split_by_group(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["WorldModelDataset", "WorldModelDataset", "WorldModelDataset"]:
        """Split by stable run/seed-qualified candidate group.

        All samples within one group go to the same split —
        no group leaks across train/val/test.

        Returns (train_ds, val_ds, test_ds).
        """
        groups = self._group_samples()
        group_ids = sorted(groups.keys())

        rng = random.Random(seed)
        rng.shuffle(group_ids)

        n = len(group_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_gids = set(group_ids[:n_train])
        val_gids = set(group_ids[n_train:n_train + n_val])
        test_gids = set(group_ids[n_train + n_val:])

        train_samples = [s for gid in train_gids for s in groups[gid]]
        val_samples = [s for gid in val_gids for s in groups[gid]]
        test_samples = [s for gid in test_gids for s in groups[gid]]

        return (
            WorldModelDataset(train_samples),
            WorldModelDataset(val_samples),
            WorldModelDataset(test_samples),
        )

    def _get_cost(
        self, sample: dict,
        risk_weight: Optional[float] = None,
        cost_lambdas: Optional[list] = None,
    ) -> float:
        """Get (possibly re-computed) cost for a sample."""
        if cost_lambdas is not None:
            fsl = sample.get("future_system_labels")
            if fsl is not None:
                return compute_realized_cost(fsl, lambdas=cost_lambdas)
        if risk_weight is None:
            return sample["realized_cost"]
        fsl = sample.get("future_system_labels")
        if fsl is None:
            return sample["realized_cost"]
        return compute_realized_cost(fsl, risk_weight=risk_weight)

    def build_pairwise_data(
        self,
        epsilon: float = 0.01,
        max_pairs: Optional[int] = None,
        seed: int = 42,
        risk_weight: Optional[float] = None,
        cost_lambdas: Optional[list] = None,
    ) -> List[dict]:
        """Build pairwise ranking pairs from samples within the same group.

        For each pair (i, j) in the same stable candidate group where
        |cost_i - cost_j| > epsilon, the lower-cost sample is sample_i (better)
        and the higher-cost sample is sample_j (worse).

        If cost_lambdas is set, re-computes cost with those per-dim weights.
        If risk_weight is set, re-computes cost with that weight for
        deadlock_risk (dim 6). Set risk_weight=0 for ablation.
        """
        groups = self._group_samples()
        pairs = []

        for members in groups.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ci = self._get_cost(members[i], risk_weight, cost_lambdas)
                    cj = self._get_cost(members[j], risk_weight, cost_lambdas)
                    if abs(ci - cj) <= epsilon:
                        continue
                    if ci < cj:
                        pairs.append({"sample_i": members[i], "sample_j": members[j]})
                    else:
                        pairs.append({"sample_i": members[j], "sample_j": members[i]})

        rng = random.Random(seed)
        rng.shuffle(pairs)

        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

        return pairs

    def summary(self) -> dict:
        """Compact summary for logging."""
        groups = self._group_samples()
        sizes = [len(m) for m in groups.values()]
        return {
            "num_samples": len(self.samples),
            "num_groups": len(groups),
            "mean_group_size": round(sum(sizes) / max(len(sizes), 1), 2),
        }
