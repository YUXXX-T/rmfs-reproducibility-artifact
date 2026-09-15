# Six-station adaptation

This study holds the 20×20 grid, 48 robots, pod zones, loads, service rules, and PP motion planner fixed while changing the station layout from four to six stations. The demand vector expands from `5+4=9` to `5+6=11`; the grid encoder remains size-agnostic.

The experiment is **adaptation, not zero-shot transfer**. Six-station core data use seeds 711–720, held-out online evaluation uses 721–730, and long-risk-label/J1-auxiliary data use 731–740. All three blocks are disjoint. Long-risk remains a model-owned output; the extra block identifies the source of its expensive continuation labels, not a separate deployed model. The J1 station predictor remains a dispatch-side auxiliary component. Each held-out seed is run at all three loads for Greedy, Hungarian, WM-Base, and ComboS1J1, producing 120 rows.

The evaluation supports a bounded generalization claim: after seed-disjoint adaptation, the hierarchy remains effective on the changed station count/layout, especially at high load. It does not establish zero-shot topology invariance, and mid-load policy ordering is not uniformly favorable.
