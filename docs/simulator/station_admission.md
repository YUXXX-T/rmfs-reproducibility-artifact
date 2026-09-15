# Station admission

The main 50-seed experiment uses `physical_only_legacy`. It enforces physical occupancy of service + queue + buffer slots, but it does **not** cap robots already admitted and traveling toward the same station. This permissive contract exposes the station-concentration failure mode studied in Fig. 5.

“Station lock” is not an internal finite-state flag. It is an operational event detected after the run from concentration of station-admission attempts. This distinction matters: an individual station may be physically full without satisfying the sustained cross-station concentration criterion, and a lock event does not itself declare a run collapsed.

Alternative committed-capacity, FIFO, and dynamic-ETA admission implementations are present in `src/WorldState/station_state.py`, but they are not enabled in the main PP comparison.
