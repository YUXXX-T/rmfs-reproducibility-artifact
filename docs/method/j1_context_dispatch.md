# J1 context dispatch

J1 orders heterogeneous fixed contexts before S1 selects robots. Its station-congestion predictor is a small shared affine-logistic head over the frozen world-model latent at each station node. It outputs two current-state channels:

- `traffic`: local stationary time, tail node density, and bottleneck-density excess;
- `service`: assigned-agent/capacity pressure and in-progress pressure.

This predictor belongs to the **J1 dispatch augmentation**. It is not the base world model's station trajectory decoder, it is not the S1 long-risk head, and training it does not update the world-model encoder. The frozen static score is:

```text
J(c) = 0.5 × predicted_service
     + 0.5 × predicted_traffic
     − service_debt(c)
```

Service debt is computed from exact pending-chain work and context age. Contexts are sorted by ascending `J` once per proposal batch; selected robots are removed from later candidate sets. The predictor checkpoint is bound to the frozen encoder and checked against its normalization contract.

The training and runtime boundary is therefore:

```text
frozen WM encoder -> station-node latent -> 2-channel J1 predictor
                                                |
exact backlog and age -> service debt ----------+-> ascending J(c)
```

Train this auxiliary component with `python scripts/train.py j1-dispatch --execute ...`. J1 does not change the world-model transition, S1 score, station admission mode, pod selection, or return policy. The predictor is implemented in `src/WorldModel/core/station_congestion_head.py`; its runtime consumer is `src/Policies/TaskAssigner/WorldModelTaskAssigner/psi_dispatch_context_assigner.py`.
