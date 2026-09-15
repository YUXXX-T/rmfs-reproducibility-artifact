"""
V5 Training Script with Validation Split
=========================================
Uses group-level train/val/test split, validation-based best checkpoint,
and optional risk_weight=0 ablation.

Usage:
    python -m WorldModel.run_train_v5
    python -m WorldModel.run_train_v5 --risk-weight 0
    python -m WorldModel.run_train_v5 --epochs 30 --max-samples 5000
"""

import argparse

import torch

from WorldModel.dataset import WorldModelDataset
from WorldModel.model import RMFSWorldModel
from WorldModel.train import train


def main():
    parser = argparse.ArgumentParser(description="V5 World Model Training")
    parser.add_argument("--data", type=str,
                        default="DataGen/wm_data/wm_train_data.pt")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts/generated/checkpoints/world_model_v5",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = use all samples")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--risk-weight", type=float, default=None,
                        help="Override deadlock_risk weight (set 0 for ablation)")
    parser.add_argument("--alpha-rank", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--max-train-pairs", type=int, default=2000)
    parser.add_argument("--max-val-pairs", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (default: auto-detect)")
    args = parser.parse_args()

    # --- Resolve device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  V5 World Model Training")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- Load data ---
    print(f"\n  Loading data from {args.data} ...")
    full_ds = WorldModelDataset.from_file(args.data)
    print(f"  Total samples: {len(full_ds)}")

    if args.max_samples > 0 and len(full_ds) > args.max_samples:
        full_ds = WorldModelDataset(full_ds.samples[:args.max_samples])
        print(f"  Truncated to {len(full_ds)} samples")

    # --- Split ---
    print(f"\n  Splitting by candidate_group_id "
          f"(train={args.train_ratio}, val={args.val_ratio}, "
          f"test={1-args.train_ratio-args.val_ratio:.2f}) ...")
    train_ds, val_ds, test_ds = full_ds.split_by_group(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    print(f"  Train: {train_ds.summary()}")
    print(f"  Val:   {val_ds.summary()}")
    print(f"  Test:  {test_ds.summary()}")

    # --- Build pairwise data ---
    rw = args.risk_weight
    if rw is not None:
        print(f"\n  risk_weight={rw} (ablation mode)")

    print(f"\n  Building pairwise data (risk_weight={rw}) ...")
    train_pairs = train_ds.build_pairwise_data(
        max_pairs=args.max_train_pairs, risk_weight=rw,
    )
    val_pairs = val_ds.build_pairwise_data(
        max_pairs=args.max_val_pairs, risk_weight=rw,
    )
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs)}")

    if len(train_pairs) == 0:
        print("  WARNING: No training pairs — ranking loss will be disabled")
    if len(val_pairs) == 0:
        print("  WARNING: No validation pairs — best checkpoint will use train loss")

    # --- Build model ---
    s0 = train_ds[0]
    node_feat_dim = s0["node_history"].shape[-1]
    edge_feat_dim = s0["edge_features"].shape[-1]
    demand_dim = s0["demand_context"].shape[0]
    action_node_dim = s0["action_node"].shape[-1]
    action_global_dim = s0["action_global"].shape[0]
    num_stations = 4
    sta = s0.get("station_node_ids")
    if sta is not None:
        num_stations = len(sta)
    horizon = s0["future_node_labels"].shape[0]

    print(f"\n  Model config: node_feat={node_feat_dim} edge_feat={edge_feat_dim} "
          f"demand={demand_dim} stations={num_stations} horizon={horizon}")

    model = RMFSWorldModel(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=64,
        demand_dim=demand_dim,
        action_node_dim=action_node_dim,
        action_global_dim=action_global_dim,
        num_stations=num_stations,
        rollout_horizon=horizon,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # --- Train ---
    print(f"\n  Training for {args.epochs} epochs ...")
    print(f"  Checkpoints -> {args.save_dir}")
    print()

    train(
        model=model,
        dataset=train_ds,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        save_name="world_model.pt",
        verbose=True,
        pairwise_data=train_pairs if train_pairs else None,
        alpha_rank=args.alpha_rank,
        margin=args.margin,
        val_dataset=val_ds,
        val_pairwise_data=val_pairs if val_pairs else None,
        risk_weight=rw,
        device=device,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
