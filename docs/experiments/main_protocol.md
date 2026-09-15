# Main 50-seed PP protocol

The frozen main protocol is:

- layout `map20_r48_s4` (20×20, 48 robots, 4 stations);
- loads low/mid/high;
- seeds 900 through 949 inclusive;
- 1,500 ticks;
- arms Greedy, Hungarian, JSQ, PhaseC (WM-Base), and ComboS1J1;
- prioritized planning with horizon 100 and goal reserve 6;
- `physical_only_legacy` station admission;
- one Greedy-recorded arrival manifest replayed exactly by all five arms.

There are 150 `(load, seed)` jobs and 750 arm-level simulations. A job is considered complete only if all five arms and the paired arrival manifest are present. The public flattened table contains 750 rows; path-bearing job JSON is intentionally not published.

The executable protocol is `src/WorldModel/evaluation/run_phase_c_physical_only_pp_50seed.py`, mirrored by `configs/evaluation/main_50seed.yaml`.
