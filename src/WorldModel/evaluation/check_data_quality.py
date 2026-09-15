"""
Data Quality Checker for World Model Training Data
====================================================
Validates tensor shapes, label statistics, and group quality.

Usage:
    python -m WorldModel.check_data_quality
    python -m WorldModel.check_data_quality --data DataGen/wm_data/wm_train_data.pt
    python -m WorldModel.check_data_quality --formal   (stricter thresholds for pre-training)
"""

import argparse
import os
import sys
from collections import Counter

import torch


SYSTEM_LABEL_NAMES = [
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
]


def main():
    parser = argparse.ArgumentParser(description="Check world model data quality")
    parser.add_argument("--data", type=str, default="DataGen/wm_data/wm_train_data.pt",
                        help="Path to wm_train_data.pt")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Cost difference threshold for pairwise pairs")
    parser.add_argument("--formal", action="store_true",
                        help="Use stricter thresholds for pre-training validation")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found")
        sys.exit(1)

    samples = torch.load(args.data, weights_only=False)
    print("=" * 60)
    print("World Model Data Quality Report")
    print("=" * 60)
    print(f"\n  Data file: {args.data}")
    print(f"  Total samples: {len(samples)}")

    if not samples:
        print("  No samples found.")
        return

    # --- Schema keys ---
    s0 = samples[0]
    print(f"\n  Schema keys: {sorted(s0.keys())}")

    # --- Tensor shapes ---
    print("\n  Tensor shapes (sample 0):")
    for key in sorted(s0.keys()):
        val = s0[key]
        if isinstance(val, torch.Tensor):
            print(f"    {key:30s} {tuple(val.shape)}")
        elif isinstance(val, (int, float)):
            print(f"    {key:30s} scalar = {val}")
        elif isinstance(val, str):
            print(f"    {key:30s} str = '{val}'")
        elif isinstance(val, dict):
            print(f"    {key:30s} dict keys = {sorted(val.keys())}")

    # --- Group statistics ---
    groups = {}
    for s in samples:
        gid = s.get("candidate_group_id", "")
        groups.setdefault(gid, []).append(s)

    sizes = [len(m) for m in groups.values()]
    size_counter = Counter(sizes)
    mean_size = sum(sizes) / max(len(sizes), 1)

    print(f"\n  Candidate groups: {len(groups)}")
    print(f"  Group size histogram: {dict(sorted(size_counter.items()))}")
    print(f"  Mean group size: {mean_size:.2f}")

    # --- Dummy count ---
    dummy_count = sum(1 for s in samples if s.get("candidate_info", {}).get("robot_id") is None)
    print(f"  Dummy samples: {dummy_count}")

    # --- Pairwise pair count ---
    pairwise_count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        costs = [m["realized_cost"] for m in members]
        for i in range(len(costs)):
            for j in range(i + 1, len(costs)):
                if abs(costs[i] - costs[j]) > args.epsilon:
                    pairwise_count += 1

    print(f"  Pairwise pairs (epsilon={args.epsilon}): {pairwise_count}")

    # --- Within-group realized_cost std ---
    cost_stds = []
    for members in groups.values():
        costs = [m["realized_cost"] for m in members]
        if len(costs) >= 2:
            mc = sum(costs) / len(costs)
            std = (sum((c - mc) ** 2 for c in costs) / len(costs)) ** 0.5
            cost_stds.append(std)

    mean_std = sum(cost_stds) / max(len(cost_stds), 1)
    print(f"  Realized cost std (mean): {mean_std:.4f}")

    # --- future_system_labels per-channel statistics ---
    all_sys = []
    for s in samples:
        fsl = s.get("future_system_labels")
        if fsl is not None:
            all_sys.append(fsl)

    sys_channel_stats = []
    if all_sys:
        stacked = torch.stack(all_sys)  # (num_samples, H, 7) — small enough
        print(f"\n  future_system_labels shape: {tuple(stacked.shape)}")
        print(f"  {'channel':45s} {'abs_sum':>12s} {'mean':>10s} {'nonzero%':>10s}")
        print(f"  {'-'*45} {'-'*12} {'-'*10} {'-'*10}")
        for dim in range(stacked.shape[-1]):
            ch = stacked[..., dim]
            abs_sum = ch.abs().sum().item()
            mean_val = ch.mean().item()
            nonzero_ratio = (ch.abs() > 1e-8).float().mean().item() * 100
            name = SYSTEM_LABEL_NAMES[dim] if dim < len(SYSTEM_LABEL_NAMES) else f"dim{dim}"
            print(f"  {name:45s} {abs_sum:12.4f} {mean_val:10.6f} {nonzero_ratio:9.1f}%")
            sys_channel_stats.append((abs_sum, mean_val, nonzero_ratio))
        del stacked

    # --- future_station_labels ---
    all_sta = []
    for s in samples:
        fstl = s.get("future_station_labels")
        if fstl is not None:
            all_sta.append(fstl)

    if all_sta:
        sta_q_abs = 0.0
        sta_q_nz = 0
        sta_l_abs = 0.0
        sta_l_nz = 0
        sta_elems = 0
        sta_shape = None
        for st in all_sta:
            if sta_shape is None:
                sta_shape = tuple(st.shape)
            q = st[..., 0]
            l = st[..., 1]
            sta_q_abs += q.abs().sum().item()
            sta_q_nz += (q.abs() > 1e-8).sum().item()
            sta_l_abs += l.abs().sum().item()
            sta_l_nz += (l.abs() > 1e-8).sum().item()
            sta_elems += q.numel()
        print(f"\n  future_station_labels shape (per sample): {sta_shape}")
        print(f"  station queue (ch0) abs_sum: {sta_q_abs:.4f}"
              f"  nonzero: {sta_q_nz / max(sta_elems, 1) * 100:.1f}%")
        print(f"  station load  (ch1) abs_sum: {sta_l_abs:.4f}"
              f"  nonzero: {sta_l_nz / max(sta_elems, 1) * 100:.1f}%")

    # --- future_node_labels (per-channel, streaming to avoid OOM) ---
    NODE_LABEL_CHANNELS = [
        "self_occupancy", "local_density", "local_wait_pressure",
        "local_blocked_pressure", "reservation_pressure", "congestion_score",
    ]
    EXPECTED_NODE_LABEL_DIM = len(NODE_LABEL_CHANNELS)  # 6

    node_ch_abs = [0.0] * EXPECTED_NODE_LABEL_DIM
    node_ch_nz = [0] * EXPECTED_NODE_LABEL_DIM
    node_ch_elems = 0
    node_shape = None
    node_dim_mismatch = False
    for s in samples:
        fnl = s.get("future_node_labels")
        if fnl is not None:
            if node_shape is None:
                node_shape = tuple(fnl.shape)
            if fnl.shape[-1] != EXPECTED_NODE_LABEL_DIM:
                node_dim_mismatch = True
                continue
            node_ch_elems += fnl[..., 0].numel()
            for ch in range(EXPECTED_NODE_LABEL_DIM):
                vals = fnl[..., ch]
                node_ch_abs[ch] += vals.abs().sum().item()
                node_ch_nz[ch] += (vals.abs() > 1e-8).sum().item()

    if node_shape is not None:
        print(f"\n  future_node_labels shape (per sample): {node_shape}")
        if node_dim_mismatch:
            print(f"  WARNING: expected last dim = {EXPECTED_NODE_LABEL_DIM}, "
                  f"got {node_shape[-1]}")
        else:
            print(f"  last dim = {node_shape[-1]} (OK)")
        if node_ch_elems > 0:
            for ch in range(EXPECTED_NODE_LABEL_DIM):
                nz_pct = node_ch_nz[ch] / node_ch_elems * 100
                print(f"  ch{ch} ({NODE_LABEL_CHANNELS[ch]:30s}) "
                      f"abs_sum: {node_ch_abs[ch]:12.4f}  "
                      f"nonzero: {nz_pct:5.1f}%")

    # --- future_mask (streaming) ---
    mask_sum = 0.0
    mask_count = 0
    for s in samples:
        fm = s.get("future_mask")
        if fm is not None:
            mask_sum += fm.sum().item()
            mask_count += fm.numel()

    valid_ratio = mask_sum / max(mask_count, 1) * 100
    if mask_count > 0:
        print(f"\n  future_mask valid ratio: {valid_ratio:.1f}%")

    # --- action_edge nonzero ratio ---
    ae_nonzero = []
    for s in samples:
        ae = s.get("action_edge")
        if ae is not None and isinstance(ae, torch.Tensor):
            ae_nonzero.append((ae.abs() > 1e-8).float().mean().item())

    if ae_nonzero:
        mean_ae = sum(ae_nonzero) / len(ae_nonzero) * 100
        print(f"  action_edge nonzero ratio: {mean_ae:.1f}%")

    # --- Within-group future_system diff count ---
    diff_count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        sys_labels = [m["future_system_labels"] for m in members if "future_system_labels" in m]
        if len(sys_labels) >= 2:
            for i in range(len(sys_labels)):
                for j in range(i + 1, len(sys_labels)):
                    if (sys_labels[i] - sys_labels[j]).abs().sum().item() > 1e-6:
                        diff_count += 1

    print(f"\n  Within-group future_system diff pairs: {diff_count}")

    # --- Validation summary ---
    mode = "FORMAL" if args.formal else "BASIC"
    print("\n" + "=" * 60)
    print(f"  Validation Checks ({mode}):")

    checks = []

    checks.append(("dummy_count == 0", dummy_count == 0, f"{dummy_count}"))

    if args.formal:
        checks.append(("mean_group_size >= 2.5", mean_size >= 2.5, f"{mean_size:.2f}"))
        checks.append(("pairwise_pairs >= 500", pairwise_count >= 500, f"{pairwise_count}"))
        checks.append(("realized_cost_std > 0.1", mean_std > 0.1, f"{mean_std:.4f}"))
        checks.append(("num_groups >= 100", len(groups) >= 100, f"{len(groups)}"))
    else:
        checks.append(("mean_group_size >= 3", mean_size >= 3.0, f"{mean_size:.2f}"))
        checks.append(("pairwise_pairs >= 1000", pairwise_count >= 1000, f"{pairwise_count}"))
        checks.append(("realized_cost_std > 0", mean_std > 0, f"{mean_std:.4f}"))

    if all_sys:
        sq_sum = sys_channel_stats[2][0] if len(sys_channel_stats) > 2 else 0.0
        checks.append(("station_queue_delta > 0", sq_sum > 0, f"{sq_sum:.4f}"))

        if args.formal:
            risk_nz = sys_channel_stats[6][2] if len(sys_channel_stats) > 6 else 100.0
            checks.append(("deadlock_risk nonzero < 99%", risk_nz < 99.0,
                           f"{risk_nz:.1f}%"))
            co_nz = sys_channel_stats[5][2] if len(sys_channel_stats) > 5 else 0.0
            checks.append(("completed_orders nonzero > 10%", co_nz > 10.0,
                           f"{co_nz:.1f}%"))

    if mask_count > 0:
        checks.append(("future_mask valid > 90%", valid_ratio > 90, f"{valid_ratio:.1f}%"))

    checks.append(("within-group diff > 0", diff_count > 0, f"{diff_count}"))

    if args.formal:
        checks.append(("total_samples >= 1000", len(samples) >= 1000, f"{len(samples)}"))

    pass_count = 0
    fail_count = 0
    warn_count = 0
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        else:
            fail_count += 1
        print(f"    [{status}] {name:40s} = {value}")

    all_pass = all(c[1] for c in checks)
    print(f"\n  Result: {pass_count} passed, {fail_count} failed")
    if all_pass:
        print("  Overall: ALL PASS — data is ready for training")
    else:
        if args.formal:
            print("  Overall: FORMAL CHECKS FAILED — fix before training")
        else:
            print("  Overall: SOME CHECKS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
