# Baselines

| Name | Context ordering | Robot selection | Learned |
|---|---|---|---|
| Greedy | proposal/FIFO order | nearest idle robot to pod by Manhattan distance | no |
| Hungarian | current batch | minimum total robot-to-pod Manhattan assignment | no |
| JSQ | ascending virtual station committed load; original proposal order tie-break | nearest idle robot; virtual load incremented after each choice | no |
| WM-Base (`PhaseC`) | fixed prefix | action-conditioned short-horizon WM score | yes |
| Proposed (`ComboS1J1`) | static J1 ascending order | S1 within-context score with tail-risk conversion | yes |

All dispatch arms receive the same fixed order/pod/station/return contexts and paired arrival stream. PP is the main downstream planner. PIBT uses strict validation and a separate planner-specific adaptation/evaluation split.

External EECBS/LNS2/LaCAM2 wrappers are included for interface completeness but are not a reported numerical dispatch baseline and their third-party source trees are not vendored.
