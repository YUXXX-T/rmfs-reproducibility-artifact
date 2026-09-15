# Temporal precedence

Fig. 5 replaces a single-seed trajectory with a 50-seed aggregate temporal view.

Panel (a) uses simulation tick on the x-axis and the fraction of the fixed 50 high-load seeds that have experienced each event by that tick on the y-axis. The denominator is always 50, including when an event is never detected. Solid curves are station-lock onset and dashed curves are paper-aligned collapse onset. The collapse curve first applies the completed-orders/deadlock endpoint label, then uses the throughput/backlog trace only to locate time.

Panel (b) shows the distribution of `collapse_tick − lock_tick` only for endpoint-collapsed runs in which both event times are available. This yields 22 Greedy pairs (median 445 ticks) and 14 ComboS1J1 pairs (median 460 ticks). All observed paired leads are positive; the plot communicates that directly without promoting a redundant inequality to a separate claim.

Mean leads are 476.8 ticks for Greedy (95% bootstrap CI [421.8, 536.8]) and 471.4 ticks for ComboS1J1 ([400.0, 557.1]). At the 0.9 lock threshold, the lead medians are 335 and 305 ticks. This is temporal mechanism evidence, not formal causal identification: the thresholds define operational events in closed-loop traces.
