from WorldModel.evaluation.phase_c_dispatch_gate1_protocol import (
    EXPECTED_PROTOCOL_SHA256,
    formal_protocol,
)
from WorldModel.evaluation.validate_phase_c_dispatch_gate1 import (
    _branch,
    _bridge_round1_contract,
    _bridge_stage1_contract,
    _liveness_audit,
)


def _arm(completed, *, deadlock=0.0):
    return {
        "order_arrival_count": 100,
        "completed_orders": completed,
        "deadlock_ratio_max": deadlock,
    }


def _metric(mean, lower=None, upper=None, *, rows=None):
    lower = mean if lower is None else lower
    upper = mean if upper is None else upper
    return {
        "overall_seed_cluster": {"mean": mean, "ci95": [lower, upper]},
        "by_load": {
            load: {"mean": mean, "ci95": [lower, upper]}
            for load in ("low", "mid", "high")
        },
        "rows": rows or [{"right_minus_left": mean}],
    }


def test_dispatch_gate1_protocol_hash_is_frozen():
    assert formal_protocol()["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256


def test_bridge_contracts_accept_values_inside_frozen_bounds():
    bridge_round1 = {
        "completion_fraction": _metric(0.01, -0.01, 0.03),
        "deadlock_ratio_max": _metric(0.0, -0.01, 0.01),
        "avg_excess_delay_relative": _metric(0.0, -0.01, 0.01),
        "completed_flow_time_relative": _metric(0.0, -0.01, 0.01),
        "open_order_fraction": _metric(0.0, -0.01, 0.01),
        "pending_order_fraction": _metric(0.0, -0.01, 0.01),
        "open_order_age_fraction": _metric(0.0, -0.01, 0.01),
        "pending_order_age_fraction": _metric(0.0, -0.01, 0.01),
        "handoff_ratio_mean": _metric(0.0, -0.005, 0.005),
        "handoff_ratio_max": _metric(0.0, -0.01, 0.01),
    }
    bridge_stage1 = {
        "completion_fraction": _metric(0.05, 0.01, 0.09),
        "deadlock_ratio_max": _metric(0.0, -0.01, 0.01),
    }
    assert _bridge_round1_contract(bridge_round1)["passed"]
    assert _bridge_stage1_contract(bridge_stage1)["passed"]


def test_liveness_audit_reuses_terminal_predicate_and_crossing_rule():
    bridge = {
        "dispatch_max_continuous_eligible_mass": 1.2,
        "max_eligible_defer_streak_bound_passed": True,
        "dispatch_crossing_violations": 0,
        "dispatch_group_span_violations": 0,
        "dispatch_group_span_max": 1.0,
        "fallback_greedy_ratio": 0.0,
        "online_robot_candidate_scope": "all_idle",
        "decision_trace_dropped": 0,
    }
    bridge_trace = [
        {
            "tick": 1100,
            "selected_action_type": "assign_robot",
            "dispatch_potential": {"crossing_reached": True},
        }
    ]
    result = _liveness_audit(bridge, bridge_trace, [])
    assert result["passed"]
    assert result["bridge_terminal"][
        "terminal_bin_robot_assignment_when_contexts_exist"
    ]


def test_diagnostic_branches_are_mutually_ordered():
    base = {
        "load": "low",
        "seed": 1,
        "arms": {
            "Stage1WorldModel": _arm(80),
            "PhaseCWorldModel": _arm(70),
            "DispatchBridge": _arm(72),
        },
    }
    passed_liveness = {"passed": True}
    failed_liveness = {"passed": False}
    assert _branch(base, failed_liveness)["branch"] == "DEFER_ABSORPTION"
    assert _branch(base, passed_liveness)["branch"] == (
        "PREEXISTING_CONGESTION_BRANCH"
    )

    induced = {
        **base,
        "arms": {
            "Stage1WorldModel": _arm(80),
            "PhaseCWorldModel": _arm(80),
            "DispatchBridge": _arm(74),
        },
    }
    assert _branch(induced, passed_liveness)["branch"] == (
        "INDUCED_CONGESTION_REGRESSION"
    )

