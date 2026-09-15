"""Assemble TD adjacent-segment tuples from online td_stream_v1 dumps (6.2 V2).

Input: ``tdstream_*.pt`` files produced by ``TDStreamProbe`` during
``evaluate_online_v6.py --td-stream-dir`` runs. Output: one
``td_stream_tuples_v1`` file with segments referencing a shared
``frames_by_tick`` table (frames stored once per run, never duplicated).

Alignment semantics (must match the offline td_tuples convention):
offline tuples record risk_seq[0] = risk after the decision tick completes
and frames[k] = state after the k-th risk record; the online stream records
risk[tau] post-step of tick tau (on_tick fires after the world step, before
advance_tick). Hence a segment starting at stride tick tau takes
risk_seq = risk@(tau+1 .. tau+K) and the bootstrap frame at tick tau+K,
matching y = sum_{k<K} gamma^k c_k + gamma^K V(s_boot).

All risk/frame access goes through an explicit tick -> index map built from
the stored tick_seq; positional identity risk_seq_full[i] == tick i is never
assumed.

Scope: this is the 6.2 V-mode on-policy TD data source only — NOT the
complete Phase C round-1 data closure (candidate ranking / counterfactual
relabeling needs the decision-point snapshot dump, a separate task).
train_td_risk_v.py consumes this schema as a pure state-V dataset. Each
segment therefore preserves the raw ``stall/deadlock/handoff`` components;
the clipped scalar ``unified_risk`` remains diagnostic only.
"""

import argparse
import datetime
import glob
import os
import re

import torch


RISK_COMPONENT_NAMES = ("stall", "deadlock", "handoff")


def _policy_family(arm_label: str) -> str:
    """Collapse load suffixes without mixing genuinely different policies."""
    label = str(arm_label or "unknown").strip()
    return re.sub(r"_(LOW|MID|HIGH)$", "", label, flags=re.IGNORECASE)


def assemble_run(stream: dict, K: int, W: int, start_stride: int,
                 min_start_tick: int) -> dict:
    tick_seq = stream["tick_seq"].tolist()
    risk_full = stream["risk_seq_full"]
    risk_components_full = stream.get("risk_components")
    if risk_components_full is None:
        raise ValueError(
            f"{stream.get('run_id', '<unknown>')}: td_stream_v1 is missing "
            "risk_components; pure TD risk V-head training requires the raw "
            "stall/deadlock/handoff ratios"
        )
    if risk_components_full.ndim != 2 or risk_components_full.shape[1] != 3:
        raise ValueError(
            f"{stream.get('run_id', '<unknown>')}: risk_components must have "
            f"shape (T, 3), got {tuple(risk_components_full.shape)}"
        )
    if len(risk_components_full) != len(tick_seq):
        raise ValueError(
            f"{stream.get('run_id', '<unknown>')}: risk component/tick length "
            "mismatch"
        )
    gate_meta = stream.get("gate_meta")
    lyapunov_summary_full = stream.get("lyapunov_l0_summary")
    productive_progress_full = stream.get("productive_progress")
    productive_progress_detail = stream.get("productive_progress_by_station")
    if lyapunov_summary_full is not None and len(lyapunov_summary_full) != len(tick_seq):
        raise ValueError(
            f"{stream.get('run_id', '<unknown>')}: L0 summary/tick length mismatch"
        )
    if productive_progress_full is not None and len(productive_progress_full) != len(tick_seq):
        raise ValueError(
            f"{stream.get('run_id', '<unknown>')}: progress/tick length mismatch"
        )
    if (productive_progress_detail is not None
            and len(productive_progress_detail) != len(tick_seq)):
        raise ValueError(
            f"{stream.get('run_id', '<unknown>')}: detailed progress/tick "
            "length mismatch"
        )
    frames_by_tick = stream["frames_by_tick"]
    run_id = stream["run_id"]
    arm_label = stream.get("arm_label", "unknown")

    tick_to_index = {int(t): i for i, t in enumerate(tick_seq)}
    frame_ticks = set(int(t) for t in frames_by_tick.keys())
    last_tick = tick_seq[-1] if tick_seq else -1

    tuples = []
    skipped_no_boot_frame = 0
    skipped_gap_in_risk = 0
    for tau in sorted(frame_ticks):
        if tau < min_start_tick:
            continue
        if tau % start_stride != 0:
            continue
        boot_tick = tau + K
        if boot_tick > last_tick:
            continue
        if boot_tick not in frame_ticks:
            skipped_no_boot_frame += 1
            continue
        # risk_seq = risk@(tau+1 .. tau+W), truncated at stream end
        idxs = []
        for t in range(tau + 1, tau + W + 1):
            i = tick_to_index.get(t)
            if i is None:
                break
            idxs.append(i)
        if len(idxs) < K:
            skipped_gap_in_risk += 1
            continue
        risk_seq = risk_full[idxs].clone()
        risk_components_seq = risk_components_full[idxs].clone()
        rec = {
            "run_id": run_id,
            "tick": tau,
            "candidate_key": f"{run_id}_t{tau}",
            "group_id": run_id,
            "r_at_start": float(risk_full[tick_to_index[tau]]),
            "risk_components_at_start": risk_components_full[
                tick_to_index[tau]
            ].clone(),
            "risk_components_at_boot": risk_components_full[
                tick_to_index[boot_tick]
            ].clone(),
            "risk_seq": risk_seq,
            "risk_components_seq": risk_components_seq,
            "s_t_tick": tau,
            "boot_tick": boot_tick,
            "truncated": len(idxs) < W,
            "mc_anchor_eligible": len(idxs) >= W,
        }
        if gate_meta is not None:
            seg = gate_meta[[tick_to_index[t] for t in range(tau + 1, boot_tick + 1)
                             if t in tick_to_index]]
            rec["gate_meta_seg_sum"] = seg.sum(dim=0)
        if lyapunov_summary_full is not None:
            rec["lyapunov_l0_at_start"] = lyapunov_summary_full[
                tick_to_index[tau]
            ].clone()
            rec["lyapunov_l0_summary_seq"] = lyapunov_summary_full[idxs].clone()
        if productive_progress_full is not None:
            rec["productive_progress_seq"] = productive_progress_full[idxs].clone()
        if productive_progress_detail is not None:
            rec["productive_progress_by_station_seq"] = [
                productive_progress_detail[index] for index in idxs
            ]
        tuples.append(rec)

    return {
        "run_id": run_id,
        "arm_label": arm_label,
        "policy_family": _policy_family(arm_label),
        "continuation_policy": f"wm_online:{arm_label}",
        "tuples": tuples,
        "frames_by_tick": frames_by_tick,
        "skipped_no_boot_frame": skipped_no_boot_frame,
        "skipped_gap_in_risk": skipped_gap_in_risk,
        "header": {k: stream.get(k) for k in (
            "schema_version", "seed", "config", "checkpoint_path",
            "energy_conv", "frame_stride", "ticks", "edge_index",
            "station_node_ids", "gate_meta_keys", "date",
            "lyapunov_l0_enabled", "lyapunov_l0_version",
            "lyapunov_l0_config", "productive_progress_names",
            "lyapunov_l0_summary_names",
        )},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream-dir", required=True,
                    help="Directory containing tdstream_*.pt files")
    ap.add_argument("--K", type=int, default=50,
                    help="TD bootstrap horizon (ticks); must divide by stride")
    ap.add_argument("--W", type=int, default=200,
                    help="Max risk_seq length recorded per segment")
    ap.add_argument("--start-stride", type=int, default=None,
                    help="Tick spacing between segment starts "
                         "(default: the stream's frame_stride)")
    ap.add_argument("--min-start-tick", type=int, default=25,
                    help="Skip segments starting before this tick "
                         "(cold counters / short history)")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--arm-label",
        action="append",
        default=None,
        help=("Exact arm_label to include. Repeat for multiple labels. "
              "Omit only when the directory already contains one policy."),
    )
    ap.add_argument(
        "--run-id-regex",
        default=None,
        help="Optional regular expression applied to stream run_id.",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.stream_dir, "tdstream_*.pt")))
    if not paths:
        raise SystemExit(f"no tdstream_*.pt under {args.stream_dir}")

    runs = []
    skipped_filter = 0
    total_tuples = 0
    total_skipped = 0
    included_paths = []
    effective_start_strides = set()
    for p in paths:
        stream = torch.load(p, map_location="cpu", weights_only=False)
        if stream.get("schema_version") != "td_stream_v1":
            raise SystemExit(f"{p}: unexpected schema "
                              f"{stream.get('schema_version')!r}")
        if args.arm_label and stream.get("arm_label") not in set(args.arm_label):
            skipped_filter += 1
            continue
        if args.run_id_regex and not re.search(
                args.run_id_regex, str(stream.get("run_id", ""))):
            skipped_filter += 1
            continue
        frame_stride = int(stream["frame_stride"])
        start_stride = args.start_stride or frame_stride
        if start_stride % frame_stride != 0:
            raise SystemExit(f"--start-stride {start_stride} must be a "
                             f"multiple of frame_stride {frame_stride}")
        if args.K % frame_stride != 0:
            raise SystemExit(f"--K {args.K} must be a multiple of "
                             f"frame_stride {frame_stride} so that "
                             f"boot_tick lands on a frame tick")
        run = assemble_run(stream, args.K, args.W, start_stride,
                           args.min_start_tick)
        included_paths.append(p)
        effective_start_strides.add(int(start_stride))
        n = len(run["tuples"])
        total_tuples += n
        total_skipped += run["skipped_no_boot_frame"]
        n_trunc = sum(1 for t in run["tuples"] if t["truncated"])
        print(f"  {os.path.basename(p)}: {n} segments "
              f"({n_trunc} truncated, "
              f"{run['skipped_no_boot_frame']} skipped_no_boot_frame)")
        runs.append(run)

    if not runs:
        raise SystemExit(
            "no streams matched the requested policy/run filters "
            f"(skipped {skipped_filter})"
        )

    policy_families = sorted({r["policy_family"] for r in runs})
    if len(policy_families) != 1:
        raise SystemExit(
            "refusing to assemble mixed-policy TD targets: "
            f"{policy_families}. Use --arm-label or --run-id-regex to build "
            "one continuation-policy family at a time."
        )
    if len(effective_start_strides) != 1:
        raise SystemExit(
            "all included streams must use one effective start stride, got "
            f"{sorted(effective_start_strides)}"
        )

    out = {
        "schema_version": "td_stream_tuples_v1",
        "K": args.K,
        "W": args.W,
        "start_stride": next(iter(effective_start_strides)),
        "min_start_tick": args.min_start_tick,
        "date": datetime.date.today().isoformat(),
        "risk_component_names": list(RISK_COMPONENT_NAMES),
        "policy_family": policy_families[0],
        "arm_labels": sorted({r["arm_label"] for r in runs}),
        "filters": {
            "arm_label": args.arm_label,
            "run_id_regex": args.run_id_regex,
        },
        "source_files": [os.path.basename(p) for p in included_paths],
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(out, args.output)
    print(f"saved {total_tuples} segments from {len(runs)} runs "
          f"({total_skipped} skipped_no_boot_frame, "
          f"{skipped_filter} filtered) -> {args.output}")


if __name__ == "__main__":
    main()
