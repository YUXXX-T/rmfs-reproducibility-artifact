from __future__ import annotations

import unittest
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from Policies.TaskAssigner.WorldModelTaskAssigner.station_context_defer_assigner import (
    StationContextDeferPipelinePhiDynamicPsiAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.station_feedback_defer_assigner import (
    StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner,
)
from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_BRAKE,
    STATION_FEEDBACK_CAUTION,
    STATION_FEEDBACK_LOCKED,
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_ACTIVE_V3,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW,
    STATION_FEEDBACK_MODE_SHADOW_V2,
    STATION_FEEDBACK_MODE_SHADOW_V3,
    STATION_FEEDBACK_OPEN,
    STATION_FEEDBACK_RECOVERY,
    STATION_FEEDBACK_RELEASE_AWARE_SCHEMA_VERSION,
    STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
    STATION_FEEDBACK_SCHEMA_VERSION,
    StationFeedbackConfig,
    StationFeedbackController,
    StationFeedbackObservation,
    build_station_feedback_observations,
)
from WorldModel.core.station_context_defer import StationDispatchSnapshot


def _config(**overrides) -> StationFeedbackConfig:
    values = {
        "service_ticks": 1,
        "caution_exit_block_streak": 1,
        "brake_exit_block_streak": 2,
        "locked_exit_block_streak": 3,
        "caution_release_drought": 2,
        "brake_release_drought": 3,
        "caution_rejection_streak": 1,
        "brake_rejection_streak": 2,
        "recovery_clear_ticks": 1,
        "recovery_release_count": 1,
        "locked_is_absorbing": True,
    }
    values.update(overrides)
    return StationFeedbackConfig(**values)


def _observation(
    tick: int,
    *,
    station_id: int = 1,
    physical=(1, 2, 3),
    capacity: int = 3,
    committed: int = 5,
    pre_pick: int = 1,
    ready: int = 1,
    pipeline: int = 7,
    exit_blocked: bool = False,
    attempts: int = 0,
    grants: int = 0,
    rejections: int = 0,
    service_agent_id: int | None = None,
    service_status: str | None = None,
    service_remaining_ticks: int | None = None,
    expected_release_in_ticks: int | None = None,
    service_release_due: bool = False,
) -> StationFeedbackObservation:
    return StationFeedbackObservation(
        station_id=station_id,
        tick=tick,
        capacity=capacity,
        physical_agent_ids=tuple(physical),
        committed_load=committed,
        pre_pick=pre_pick,
        ready_unadmitted=ready,
        pipeline_mass=pipeline,
        exit_blocked=exit_blocked,
        admission_attempts=attempts,
        admission_grants=grants,
        admission_rejections=rejections,
        service_agent_id=service_agent_id,
        service_status=service_status,
        service_remaining_ticks=service_remaining_ticks,
        expected_release_in_ticks=expected_release_in_ticks,
        service_release_due=service_release_due,
    )


class StationFeedbackControllerTests(unittest.TestCase):
    def test_service_tick_defaults_are_ordered_and_untuned(self):
        config = StationFeedbackConfig.for_service_ticks(5)
        self.assertEqual(config.caution_exit_block_streak, 5)
        self.assertEqual(config.brake_exit_block_streak, 8)
        self.assertEqual(config.locked_exit_block_streak, 10)
        self.assertEqual(config.caution_release_drought, 12)
        self.assertEqual(config.brake_release_drought, 18)
        self.assertFalse(config.early_brake_enabled)
        self.assertEqual(config.early_brake_committed_multiplier, 1.4)
        self.assertEqual(config.early_brake_pressure_streak, 3)
        self.assertEqual(config.early_brake_exit_block_streak, 3)
        self.assertEqual(config.early_brake_release_drought, 5)
        self.assertEqual(config.early_brake_rejection_streak, 3)
        self.assertEqual(config.early_brake_evidence_required, 2)
        self.assertNotIn("release_signal", config.as_dict())

    def test_service_due_profile_redefines_drought_as_overdue_time(self):
        config = StationFeedbackConfig.for_service_due_release(
            5, early_brake_enabled=True
        )
        self.assertEqual(
            config.release_signal,
            STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
        )
        self.assertEqual(config.caution_release_drought, 2)
        self.assertEqual(config.brake_release_drought, 5)
        self.assertEqual(config.early_brake_release_drought, 2)
        self.assertEqual(config.release_handoff_grace_ticks, 1)
        self.assertEqual(
            config.schema_version,
            STATION_FEEDBACK_RELEASE_AWARE_SCHEMA_VERSION,
        )
        self.assertEqual(
            config.as_dict()["release_signal"],
            STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
        )

    def test_v1_profile_remains_late_brake_under_committed_pressure(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_ticks(5)
        )
        snapshot = None
        for tick in range(5):
            snapshot = controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                exit_blocked=True,
            )])[1]

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.state, STATION_FEEDBACK_CAUTION)
        self.assertFalse(snapshot.early_brake_condition)
        self.assertFalse(snapshot.committed_pressure)
        self.assertEqual(snapshot.committed_pressure_streak, 0)

    def test_v2_combined_pressure_brakes_at_caution_horizon(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_ticks(
                5, early_brake_enabled=True
            )
        )
        snapshot = None
        for tick in range(5):
            snapshot = controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                exit_blocked=True,
            )])[1]

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(snapshot.dominant_reason, "early_committed_pressure")
        self.assertEqual(snapshot.committed_pressure_threshold, 10)
        self.assertEqual(snapshot.committed_pressure_streak, 5)
        self.assertEqual(snapshot.early_brake_evidence_count, 2)
        self.assertTrue(snapshot.early_exit_block_evidence)
        self.assertTrue(snapshot.early_release_drought_evidence)
        self.assertFalse(snapshot.early_rejection_evidence)
        self.assertEqual(snapshot.schema_version, STATION_FEEDBACK_SCHEMA_VERSION)
        self.assertNotIn("release_overdue_ticks", snapshot.as_dict())

    def test_v2_committed_pressure_alone_never_brakes(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_ticks(
                5, early_brake_enabled=True
            )
        )
        snapshots = []
        for tick in range(7):
            snapshots.append(controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                exit_blocked=False,
            )])[1])

        self.assertTrue(all(
            snapshot.state != STATION_FEEDBACK_BRAKE
            for snapshot in snapshots
        ))
        self.assertEqual(snapshots[-1].early_brake_evidence_count, 1)
        self.assertFalse(snapshots[-1].early_brake_condition)

    def test_v2_observation_gap_resets_committed_pressure_streak(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_ticks(
                5, early_brake_enabled=True
            )
        )
        controller.update([_observation(
            0,
            physical=tuple(range(7)),
            capacity=7,
            committed=10,
            pipeline=17,
            exit_blocked=True,
        )])
        controller.update([_observation(
            1,
            physical=tuple(range(7)),
            capacity=7,
            committed=10,
            pipeline=17,
            exit_blocked=True,
        )])
        after_gap = controller.update([_observation(
            5,
            physical=tuple(range(7)),
            capacity=7,
            committed=10,
            pipeline=17,
            exit_blocked=True,
        )])[1]

        self.assertEqual(after_gap.observation_gap_ticks, 3)
        self.assertEqual(after_gap.committed_pressure_streak, 1)
        self.assertFalse(after_gap.early_brake_condition)

    def test_v2_early_brake_has_time_to_observe_release_recovery(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_ticks(
                5, early_brake_enabled=True
            )
        )
        for tick in range(5):
            brake = controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                exit_blocked=True,
            )])[1]
        recovery = controller.update([_observation(
            5,
            physical=tuple(range(1, 7)),
            capacity=7,
            committed=9,
            pipeline=15,
            exit_blocked=False,
        )])[1]

        self.assertEqual(brake.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(brake.exit_blocked_streak, 5)
        self.assertEqual(recovery.release_delta, 1)
        self.assertEqual(recovery.state, STATION_FEEDBACK_RECOVERY)

    def test_v3_normal_service_cycle_is_not_release_drought(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_due_release(
                5, early_brake_enabled=True
            )
        )
        snapshots = []
        for tick, remaining in enumerate((5, 4, 3, 2, 1)):
            snapshots.append(controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                exit_blocked=True,
                attempts=tick,
                rejections=tick,
                service_agent_id=0,
                service_status="delivering",
                service_remaining_ticks=remaining,
                expected_release_in_ticks=remaining + 1,
            )])[1])
        due = controller.update([_observation(
            5,
            physical=tuple(range(7)),
            capacity=7,
            committed=10,
            pipeline=17,
            exit_blocked=True,
            attempts=5,
            rejections=5,
            service_agent_id=0,
            service_status="exiting",
            service_remaining_ticks=0,
            expected_release_in_ticks=0,
            service_release_due=True,
        )])[1]
        released = controller.update([_observation(
            6,
            physical=tuple(range(1, 7)),
            capacity=7,
            committed=9,
            pipeline=15,
            exit_blocked=False,
            attempts=5,
            rejections=5,
        )])[1]

        self.assertEqual(snapshots[-1].release_drought, 5)
        self.assertEqual(snapshots[-1].release_overdue_ticks, 0)
        self.assertEqual(snapshots[-1].control_release_drought, 0)
        self.assertTrue(all(
            snapshot.state != STATION_FEEDBACK_BRAKE
            for snapshot in snapshots
        ))
        self.assertEqual(due.release_due_without_progress_streak, 1)
        self.assertEqual(due.release_overdue_ticks, 0)
        self.assertNotEqual(due.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(released.release_delta, 1)
        self.assertEqual(released.release_overdue_ticks, 0)

    def test_v3_rejection_and_busy_time_without_due_release_never_brake(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_due_release(
                5, early_brake_enabled=True
            )
        )
        snapshots = []
        for tick in range(12):
            snapshots.append(controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                attempts=10 * tick,
                rejections=10 * tick,
                service_agent_id=0,
                service_status="delivering",
                service_remaining_ticks=5,
                expected_release_in_ticks=6,
            )])[1])

        self.assertGreaterEqual(snapshots[-1].release_drought, 5)
        self.assertGreaterEqual(snapshots[-1].rejection_tick_streak, 3)
        self.assertEqual(snapshots[-1].release_overdue_ticks, 0)
        self.assertEqual(snapshots[-1].state, STATION_FEEDBACK_OPEN)
        self.assertTrue(all(
            not snapshot.early_brake_condition for snapshot in snapshots
        ))

    def test_v3_missed_due_release_triggers_guarded_early_brake(self):
        controller = StationFeedbackController(
            StationFeedbackConfig.for_service_due_release(
                5, early_brake_enabled=True
            )
        )
        snapshots = []
        for tick in range(3):
            snapshots.append(controller.update([_observation(
                tick,
                physical=tuple(range(7)),
                capacity=7,
                committed=10,
                pipeline=17,
                exit_blocked=True,
                service_agent_id=0,
                service_status="exiting",
                service_remaining_ticks=0,
                expected_release_in_ticks=0,
                service_release_due=True,
            )])[1])

        brake = snapshots[-1]
        self.assertEqual(brake.release_due_without_progress_streak, 3)
        self.assertEqual(brake.release_overdue_ticks, 2)
        self.assertEqual(brake.due_exit_blocked_streak, 3)
        self.assertTrue(brake.early_exit_block_evidence)
        self.assertTrue(brake.early_release_drought_evidence)
        self.assertTrue(brake.early_brake_condition)
        self.assertEqual(brake.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(
            brake.dominant_reason, "early_service_release_overdue"
        )

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            _config(brake_exit_block_streak=1)

    def test_exit_warning_brakes_before_absorbing_lock(self):
        controller = StationFeedbackController(_config())
        first = controller.update([
            _observation(0, exit_blocked=True)
        ])[1]
        second = controller.update([
            _observation(1, exit_blocked=True)
        ])[1]
        third = controller.update([
            _observation(2, exit_blocked=True)
        ])[1]

        self.assertEqual(first.state, STATION_FEEDBACK_CAUTION)
        self.assertEqual(second.state, STATION_FEEDBACK_BRAKE)
        self.assertTrue(second.defer_new_context)
        self.assertEqual(third.state, STATION_FEEDBACK_LOCKED)
        self.assertEqual(third.dominant_reason, "persistent_exit_block")

        cleared = controller.update([
            _observation(3, exit_blocked=False, physical=(1, 2))
        ])[1]
        self.assertEqual(cleared.state, STATION_FEEDBACK_LOCKED)
        self.assertTrue(cleared.defer_new_context)

    def test_healthy_handoff_occupancy_with_release_does_not_lock(self):
        controller = StationFeedbackController(_config())
        first = controller.update([
            _observation(0, physical=(1, 2, 3), exit_blocked=True)
        ])[1]
        flowing = controller.update([
            _observation(1, physical=(2, 3, 4), exit_blocked=True)
        ])[1]

        self.assertTrue(first.exit_blocked_without_release)
        self.assertEqual(first.exit_blocked_streak, 1)
        self.assertEqual(flowing.release_delta, 1)
        self.assertFalse(flowing.exit_blocked_without_release)
        self.assertEqual(flowing.exit_blocked_streak, 0)
        self.assertEqual(flowing.state, STATION_FEEDBACK_OPEN)

    def test_release_drought_can_trigger_preventive_brake(self):
        controller = StationFeedbackController(_config(
            caution_exit_block_streak=20,
            brake_exit_block_streak=21,
            locked_exit_block_streak=22,
        ))
        states = []
        for tick in range(3):
            states.append(controller.update([_observation(tick)])[1])
        self.assertEqual(states[0].state, STATION_FEEDBACK_OPEN)
        self.assertEqual(states[1].state, STATION_FEEDBACK_CAUTION)
        self.assertEqual(states[2].state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(states[2].dominant_reason, "release_drought")

    def test_rejection_streak_counts_ticks_not_raw_attempts(self):
        controller = StationFeedbackController(_config(
            caution_exit_block_streak=20,
            brake_exit_block_streak=21,
            locked_exit_block_streak=22,
            caution_release_drought=20,
            brake_release_drought=21,
        ))
        controller.update([_observation(0, physical=(1,), attempts=0)])
        caution = controller.update([_observation(
            1,
            physical=(1,),
            attempts=100,
            rejections=100,
        )])[1]
        brake = controller.update([_observation(
            2,
            physical=(1,),
            attempts=200,
            rejections=200,
        )])[1]

        self.assertEqual(caution.rejection_tick_streak, 1)
        self.assertEqual(caution.state, STATION_FEEDBACK_CAUTION)
        self.assertEqual(brake.rejection_tick_streak, 2)
        self.assertEqual(brake.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(brake.dominant_reason, "rejection_streak")

    def test_unobserved_tick_gap_does_not_fabricate_a_streak(self):
        controller = StationFeedbackController(_config(
            caution_exit_block_streak=2,
            brake_exit_block_streak=3,
            locked_exit_block_streak=4,
            caution_release_drought=20,
            brake_release_drought=21,
            caution_rejection_streak=20,
            brake_rejection_streak=21,
        ))
        first = controller.update([
            _observation(0, exit_blocked=True)
        ])[1]
        after_gap = controller.update([
            _observation(5, exit_blocked=True)
        ])[1]

        self.assertEqual(first.exit_blocked_streak, 1)
        self.assertEqual(after_gap.exit_blocked_streak, 1)
        self.assertEqual(after_gap.observation_gap_ticks, 4)
        self.assertEqual(after_gap.state, STATION_FEEDBACK_OPEN)

    def test_brake_latch_has_an_explicit_audit_reason(self):
        controller = StationFeedbackController(_config(
            caution_release_drought=20,
            brake_release_drought=21,
            caution_rejection_streak=20,
            brake_rejection_streak=21,
        ))
        controller.update([_observation(0, exit_blocked=True)])
        controller.update([_observation(1, exit_blocked=True)])
        latched = controller.update([
            _observation(2, exit_blocked=False)
        ])[1]

        self.assertEqual(latched.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(
            latched.dominant_reason,
            "brake_waiting_for_observed_release",
        )

    def test_observed_release_moves_brake_through_recovery(self):
        controller = StationFeedbackController(_config())
        controller.update([_observation(0, exit_blocked=True)])
        brake = controller.update([
            _observation(1, exit_blocked=True)
        ])[1]
        recovery = controller.update([_observation(
            2,
            exit_blocked=False,
            physical=(1, 2),
            committed=4,
            pipeline=5,
        )])[1]
        opened = controller.update([_observation(
            3,
            exit_blocked=False,
            physical=(1,),
            committed=2,
            pre_pick=0,
            ready=0,
            pipeline=2,
        )])[1]

        self.assertEqual(brake.state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(recovery.release_delta, 1)
        self.assertEqual(recovery.state, STATION_FEEDBACK_RECOVERY)
        self.assertEqual(opened.state, STATION_FEEDBACK_OPEN)

    def test_station_states_are_independent(self):
        controller = StationFeedbackController(_config())
        controller.update([
            _observation(0, station_id=1, exit_blocked=True),
            _observation(
                0,
                station_id=2,
                physical=(),
                committed=0,
                pre_pick=0,
                ready=0,
                pipeline=0,
            ),
        ])
        result = controller.update([
            _observation(1, station_id=1, exit_blocked=True),
            _observation(
                1,
                station_id=2,
                physical=(),
                committed=0,
                pre_pick=0,
                ready=0,
                pipeline=0,
            ),
        ])
        self.assertEqual(result[1].state, STATION_FEEDBACK_BRAKE)
        self.assertEqual(result[2].state, STATION_FEEDBACK_OPEN)

    def test_observation_builder_requires_service_and_exit_occupancy(self):
        class Queue:
            capacity = 3
            exit_position = (0, 1)
            service = SimpleNamespace(agent_id=1)

            @staticmethod
            def physical_agent_ids():
                return {1, 2, 3}

            @staticmethod
            def admission_metrics():
                return {"attempts": 7, "granted": 5, "rejected_capacity": 2}

        agent = SimpleNamespace(agent_id=9, position=(0, 1))
        world = SimpleNamespace(
            tick=4,
            agents=[agent],
            station_state=SimpleNamespace(
                get_queue=lambda station_id: Queue()
            ),
        )
        station_snapshot = SimpleNamespace(
            capacity=3,
            committed_load=5,
            pre_pick=1,
            ready_unadmitted=1,
            pipeline_mass=7,
        )
        blocked = build_station_feedback_observations(
            world, {1: station_snapshot}
        )[0]
        self.assertTrue(blocked.exit_blocked)

        Queue.service.agent_id = None
        clear = build_station_feedback_observations(
            world, {1: station_snapshot}
        )[0]
        self.assertFalse(clear.exit_blocked)

    def test_observation_builder_projects_actual_service_release_phase(self):
        class Queue:
            exit_position = (0, 1)
            service = SimpleNamespace(agent_id=1)

            @staticmethod
            def physical_agent_ids():
                return {1, 2, 3}

            @staticmethod
            def admission_metrics():
                return {"attempts": 0, "granted": 0, "rejected_capacity": 0}

        service_agent = SimpleNamespace(
            agent_id=1,
            position=(0, 0),
            status=SimpleNamespace(name="DELIVERING"),
            wait_ticks=3,
        )
        world = SimpleNamespace(
            tick=4,
            agents=[service_agent],
            get_agent=lambda agent_id: service_agent,
            station_state=SimpleNamespace(get_queue=lambda station_id: Queue()),
        )
        station_snapshot = SimpleNamespace(
            capacity=3,
            committed_load=5,
            pre_pick=1,
            ready_unadmitted=1,
            pipeline_mass=7,
        )

        delivering = build_station_feedback_observations(
            world, {1: station_snapshot}, service_ticks=5
        )[0]
        self.assertEqual(delivering.service_agent_id, 1)
        self.assertEqual(delivering.service_status, "delivering")
        self.assertEqual(delivering.service_remaining_ticks, 3)
        self.assertEqual(delivering.expected_release_in_ticks, 4)
        self.assertFalse(delivering.service_release_due)

        service_agent.status = SimpleNamespace(name="EXITING")
        service_agent.wait_ticks = 0
        exiting = build_station_feedback_observations(
            world, {1: station_snapshot}, service_ticks=5
        )[0]
        self.assertEqual(exiting.expected_release_in_ticks, 0)
        self.assertTrue(exiting.service_release_due)


class StationFeedbackAssignerIsolationTests(unittest.TestCase):
    @staticmethod
    def _brake_snapshot():
        controller = StationFeedbackController(_config())
        controller.update([_observation(0, exit_blocked=True)])
        return controller.update([_observation(1, exit_blocked=True)])[1]

    @staticmethod
    def _assigner(mode: str):
        assigner = (
            StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner.__new__(
                StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner
            )
        )
        assigner._station_feedback_mode = mode
        assigner._station_feedback_current = {
            1: StationFeedbackAssignerIsolationTests._brake_snapshot()
        }
        assigner._station_feedback_batch_filter_tick = None
        assigner._station_feedback_batch_filter_stations_this_tick = set()
        assigner._station_feedback_batch_filter_contexts_this_tick = set()
        assigner._station_feedback_batch_filter_unique_contexts = set()
        assigner.stats = {
            "station_feedback_context_evaluations": 0,
            "station_feedback_would_defer_evaluations": 0,
            "station_feedback_forced_defers": 0,
            "station_feedback_selected_contexts": 0,
            "station_feedback_contexts_materialized_in_defer_state": 0,
            "station_feedback_batch_filter_batches": 0,
            "station_feedback_batch_filter_contexts_scanned": 0,
            "station_feedback_batch_filter_would_suppress_batches": 0,
            "station_feedback_batch_filter_would_suppress_station_ticks": 0,
            "station_feedback_batch_filter_would_suppress_context_ticks": 0,
            "station_feedback_batch_filter_suppressed_batches": 0,
            "station_feedback_batch_filter_suppressed_station_ticks": 0,
            "station_feedback_batch_filter_suppressed_context_ticks": 0,
            "station_feedback_batch_filter_duplicate_station_tick_records_suppressed": 0,
            "station_feedback_batch_filter_duplicate_context_tick_records_suppressed": 0,
            "station_context_defer_decisions": 0,
        }
        return assigner

    def test_frozen_parent_hooks_are_noops(self):
        parent = StationContextDeferPipelinePhiDynamicPsiAssigner.__new__(
            StationContextDeferPipelinePhiDynamicPsiAssigner
        )
        self.assertIsNone(
            parent._station_context_feedback_for_row(
                None, {"station_id": 1}, None
            )
        )
        self.assertIsNone(
            parent._record_station_context_feedback_outcome(None, "selected")
        )
        context = SimpleNamespace(order_id=1, pod_id=2, station_id=3)
        remaining = [(0, context)]
        filtered, steps = parent._station_context_feedback_batch_filter(
            SimpleNamespace(tick=1), remaining, {}
        )
        self.assertEqual(filtered, remaining)
        self.assertEqual(steps, [])

    def test_off_mode_does_not_attach_feedback(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_OFF)
        self.assertIsNone(
            assigner._station_context_feedback_for_row(
                None, {"station_id": 1}, None
            )
        )

    def test_mapping_overrides_keep_service_scaled_defaults(self):
        def fake_parent_init(instance, **kwargs):
            instance.stats = {}

        with patch.object(
            StationContextDeferPipelinePhiDynamicPsiAssigner,
            "__init__",
            fake_parent_init,
        ):
            assigner = StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner(
                station_feedback_mode=STATION_FEEDBACK_MODE_OFF,
                station_feedback_service_ticks=7,
                station_feedback_config={"recovery_release_count": 3},
            )

        config = assigner._station_feedback_config
        self.assertEqual(config.service_ticks, 7)
        self.assertEqual(config.caution_exit_block_streak, 7)
        self.assertEqual(config.recovery_release_count, 3)

    def test_negative_transition_trace_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner(
                station_feedback_trace_max_records=-1
            )

    def test_off_mode_preparation_is_counted_once_per_tick(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_OFF)
        assigner._station_feedback_last_prepared_tick = None
        assigner.stats["station_feedback_off_batches"] = 0
        world = SimpleNamespace(tick=4)

        assigner._prepare_station_context_feedback(world, {})
        assigner._prepare_station_context_feedback(world, {})

        self.assertEqual(assigner.stats["station_feedback_off_batches"], 1)

    def test_shadow_mode_audits_but_does_not_defer(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_SHADOW)
        feedback = assigner._station_context_feedback_for_row(
            None, {"station_id": 1}, None
        )
        self.assertTrue(feedback["would_defer"])
        self.assertFalse(feedback["defer_context"])
        assigner._record_station_context_feedback_outcome(
            feedback, "selected"
        )
        self.assertEqual(
            assigner.stats[
                "station_feedback_contexts_materialized_in_defer_state"
            ],
            1,
        )

    def test_shadow_batch_filter_audits_without_removing_contexts(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_SHADOW)
        assigner.stats = defaultdict(int, assigner.stats)
        context = SimpleNamespace(order_id=10, pod_id=20, station_id=1)
        remaining = [(0, context)]
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=3,
            committed_in_transit=0,
            pre_pick=2,
            ready_unadmitted=1,
        )

        filtered, steps = assigner._station_context_feedback_batch_filter(
            SimpleNamespace(tick=1),
            remaining,
            {1: station_snapshot},
        )

        self.assertEqual(filtered, remaining)
        self.assertEqual(steps, [])
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_would_suppress_station_ticks"
            ],
            1,
        )
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_would_suppress_context_ticks"
            ],
            1,
        )
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_suppressed_context_ticks"
            ],
            0,
        )
        self.assertEqual(assigner.stats["station_feedback_forced_defers"], 0)

    def test_v2_shadow_uses_the_same_non_mutating_filter_contract(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_SHADOW_V2)
        assigner.stats = defaultdict(int, assigner.stats)
        context = SimpleNamespace(order_id=10, pod_id=20, station_id=1)
        remaining = [(0, context)]
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=3,
            committed_in_transit=0,
            pre_pick=2,
            ready_unadmitted=1,
        )

        filtered, steps = assigner._station_context_feedback_batch_filter(
            SimpleNamespace(tick=1), remaining, {1: station_snapshot}
        )

        self.assertEqual(filtered, remaining)
        self.assertEqual(steps, [])
        self.assertEqual(assigner.stats["station_feedback_forced_defers"], 0)

    def test_active_mode_forces_station_local_context_defer(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_ACTIVE)
        feedback = assigner._station_context_feedback_for_row(
            None, {"station_id": 1}, None
        )
        self.assertTrue(feedback["would_defer"])
        self.assertTrue(feedback["defer_context"])
        assigner._record_station_context_feedback_outcome(
            feedback, "station_feedback_deferred_context"
        )
        self.assertEqual(
            assigner.stats["station_feedback_forced_defers"], 1
        )

    def test_active_batch_filter_deduplicates_station_tick_records(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_ACTIVE)
        assigner.stats = defaultdict(int, assigner.stats)
        contexts = [
            SimpleNamespace(order_id=10, pod_id=20, station_id=1),
            SimpleNamespace(order_id=11, pod_id=21, station_id=1),
        ]
        remaining = list(enumerate(contexts))
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=3,
            committed_in_transit=0,
            pre_pick=2,
            ready_unadmitted=1,
        )
        world = SimpleNamespace(tick=1)

        first, first_steps = assigner._station_context_feedback_batch_filter(
            world, remaining, {1: station_snapshot}
        )
        second, second_steps = assigner._station_context_feedback_batch_filter(
            world, remaining, {1: station_snapshot}
        )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(len(first_steps), 1)
        self.assertEqual(second_steps, [])
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_suppressed_station_ticks"
            ],
            1,
        )
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_suppressed_context_ticks"
            ],
            2,
        )
        self.assertEqual(assigner.stats["station_feedback_forced_defers"], 2)
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_duplicate_station_tick_records_suppressed"
            ],
            1,
        )
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_duplicate_context_tick_records_suppressed"
            ],
            2,
        )
        self.assertEqual(
            len(assigner._station_feedback_batch_filter_unique_contexts), 2
        )

    def test_v2_active_reuses_station_local_batch_deduplication(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_ACTIVE_V2)
        assigner.stats = defaultdict(int, assigner.stats)
        contexts = [
            SimpleNamespace(order_id=10, pod_id=20, station_id=1),
            SimpleNamespace(order_id=11, pod_id=21, station_id=1),
        ]
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=3,
            committed_in_transit=0,
            pre_pick=2,
            ready_unadmitted=1,
        )

        filtered, steps = assigner._station_context_feedback_batch_filter(
            SimpleNamespace(tick=1),
            list(enumerate(contexts)),
            {1: station_snapshot},
        )

        self.assertEqual(filtered, [])
        self.assertEqual(len(steps), 1)
        self.assertEqual(assigner.stats["station_feedback_forced_defers"], 2)

    def test_v3_modes_reuse_the_isolated_station_local_filter(self):
        context = SimpleNamespace(order_id=10, pod_id=20, station_id=1)
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=3,
            committed_in_transit=0,
            pre_pick=2,
            ready_unadmitted=1,
        )

        shadow = self._assigner(STATION_FEEDBACK_MODE_SHADOW_V3)
        shadow.stats = defaultdict(int, shadow.stats)
        shadow_filtered, _ = shadow._station_context_feedback_batch_filter(
            SimpleNamespace(tick=1),
            [(0, context)],
            {1: station_snapshot},
        )
        self.assertEqual(shadow_filtered, [(0, context)])
        self.assertEqual(shadow.stats["station_feedback_forced_defers"], 0)

        active = self._assigner(STATION_FEEDBACK_MODE_ACTIVE_V3)
        active.stats = defaultdict(int, active.stats)
        active_filtered, steps = active._station_context_feedback_batch_filter(
            SimpleNamespace(tick=1),
            [(0, context)],
            {1: station_snapshot},
        )
        self.assertEqual(active_filtered, [])
        self.assertEqual(len(steps), 1)
        self.assertEqual(active.stats["station_feedback_forced_defers"], 1)

    def test_active_defer_is_integrated_and_pauses_liveness_debt(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_ACTIVE)
        assigner.stats = defaultdict(int, assigner.stats)
        assigner._station_context_defer_mass = {}
        assigner._station_context_defer_ticks = {}
        assigner._station_context_defer_bound = {}
        assigner._station_context_defer_clock = {}
        assigner._station_context_defer_last_update_tick = {}
        assigner._station_feedback_last_prepared_tick = 1
        assigner._psi_last_final_budget = 1
        assigner._psi_distance = None
        assigner._psi_durations = None
        assigner._dynamic_reserved_robot_ids = set()
        assigner.dynamic_trace_enabled = False
        assigner.dynamic_probe_trace_records = []
        assigner.dynamic_trace_max_records = 0
        assigner._prune_defer_mass = lambda world: None
        assigner._initial_pending_counts = lambda debt: {1: 1}
        assigner._sort_rows = lambda rows: list(rows)
        assigner._row_key = lambda row: (10, 20, 1)
        assigner._row_text = lambda row: "order=10,pod=20,station=1"
        assigner._append_dynamic_trace = lambda record: None

        context = SimpleNamespace(order_id=10, pod_id=20, station_id=1)
        row = {
            "original_index": 0,
            "order_id": 10,
            "pod_id": 20,
            "station_id": 1,
            "context": context,
            "service": 0.2,
            "traffic": 0.2,
            "service_debt": 0.9,
            "backlog_score": 0.9,
            "age_score": 0.9,
            "free_flow_time": 5.0,
            "j_score": -0.7,
        }
        dynamic_rows_called = []

        def dynamic_rows(*args, **kwargs):
            dynamic_rows_called.append(True)
            return [dict(row)]

        assigner._dynamic_rows = dynamic_rows
        world = SimpleNamespace(
            tick=1,
            get_idle_agents=lambda: [SimpleNamespace(agent_id=7)],
        )
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=3,
            committed_in_transit=0,
            pre_pick=2,
            ready_unadmitted=1,
        )
        contexts = [context]
        module = (
            "Policies.TaskAssigner.WorldModelTaskAssigner."
            "station_context_defer_assigner"
        )
        with patch(
            f"{module}.build_context_debt_features", return_value={}
        ), patch(
            f"{module}.build_station_dispatch_snapshots",
            return_value={1: station_snapshot},
        ):
            choices = assigner._select_robots_interleaved(
                world,
                contexts,
                {1: {"service": 0.2, "traffic": 0.2}},
            )

        self.assertEqual(choices, {})
        self.assertEqual(contexts, [])
        self.assertEqual(
            assigner.stats["station_feedback_forced_defers"], 1
        )
        self.assertEqual(dynamic_rows_called, [])
        self.assertEqual(assigner.stats["dynamic_probe_steps"], 1)
        self.assertEqual(assigner.stats["station_context_defer_steps"], 1)
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_suppressed_station_ticks"
            ],
            1,
        )
        self.assertNotIn((10, 20, 1), assigner._station_context_defer_mass)
        self.assertNotIn((10, 20, 1), assigner._station_context_defer_ticks)

    def test_active_brake_continues_with_a_healthy_station_context(self):
        assigner = self._assigner(STATION_FEEDBACK_MODE_ACTIVE)
        healthy = StationFeedbackController(_config()).update([
            _observation(
                0,
                station_id=2,
                physical=(),
                committed=0,
                pre_pick=0,
                ready=0,
                pipeline=0,
            )
        ])[2]
        assigner._station_feedback_current[2] = healthy
        assigner.stats = defaultdict(int, assigner.stats)
        assigner._station_context_defer_mass = {}
        assigner._station_context_defer_ticks = {}
        assigner._station_context_defer_bound = {}
        assigner._station_context_defer_clock = {}
        assigner._station_context_defer_last_update_tick = {}
        assigner._station_feedback_last_prepared_tick = 1
        assigner._psi_last_final_budget = 1
        assigner._psi_distance = None
        assigner._psi_durations = None
        assigner._dynamic_reserved_robot_ids = set()
        assigner.dynamic_trace_enabled = False
        assigner.dynamic_probe_trace_records = []
        assigner.dynamic_trace_max_records = 0
        assigner._prune_defer_mass = lambda world: None
        assigner._initial_pending_counts = lambda debt: {1: 1, 2: 1}
        assigner._sort_rows = lambda rows: list(rows)
        assigner._row_key = lambda row: (
            int(row["order_id"]),
            int(row["pod_id"]),
            int(row["station_id"]),
        )
        assigner._row_text = lambda row: (
            f"order={row['order_id']},pod={row['pod_id']},"
            f"station={row['station_id']}"
        )
        assigner._virtual_idle_view = lambda world, idle: nullcontext()
        assigner._append_dynamic_trace = lambda record: None

        blocked_context = SimpleNamespace(
            order_id=10, pod_id=20, station_id=1
        )
        blocked_context_2 = SimpleNamespace(
            order_id=12, pod_id=22, station_id=1
        )
        healthy_context = SimpleNamespace(
            order_id=11, pod_id=21, station_id=2
        )
        rows = {
            1: {
                "order_id": 10,
                "pod_id": 20,
                "station_id": 1,
                "context": blocked_context,
                "service": 0.2,
                "traffic": 0.2,
                "service_debt": 0.9,
                "backlog_score": 0.9,
                "age_score": 0.9,
                "free_flow_time": 5.0,
                "j_score": -0.8,
            },
            2: {
                "order_id": 11,
                "pod_id": 21,
                "station_id": 2,
                "context": healthy_context,
                "service": 0.2,
                "traffic": 0.2,
                "service_debt": 0.9,
                "backlog_score": 0.9,
                "age_score": 0.9,
                "free_flow_time": 5.0,
                "j_score": -0.7,
            },
        }

        dynamic_row_station_sets = []

        def dynamic_rows(_world, remaining, *_args, **_kwargs):
            dynamic_row_station_sets.append({
                int(context.station_id) for _index, context in remaining
            })
            return [
                {
                    **rows[int(context.station_id)],
                    "original_index": int(original_index),
                }
                for original_index, context in remaining
            ]

        assigner._dynamic_rows = dynamic_rows
        world = SimpleNamespace(
            tick=1,
            get_idle_agents=lambda: [SimpleNamespace(agent_id=7)],
        )
        station_snapshots = {
            1: StationDispatchSnapshot(
                station_id=1,
                capacity=3,
                physical=3,
                committed_in_transit=0,
                pre_pick=2,
                ready_unadmitted=1,
            ),
            2: StationDispatchSnapshot(
                station_id=2,
                capacity=3,
                physical=0,
                committed_in_transit=0,
                pre_pick=0,
                ready_unadmitted=0,
            ),
        }
        contexts = [blocked_context, blocked_context_2, healthy_context]
        debt_feature_context_batches = []

        def debt_features(_world, context_batch, *_args, **_kwargs):
            debt_feature_context_batches.append(list(context_batch))
            return {}

        module = (
            "Policies.TaskAssigner.WorldModelTaskAssigner."
            "station_context_defer_assigner"
        )
        with patch(
            f"{module}.build_context_debt_features",
            side_effect=debt_features,
        ), patch(
            f"{module}.build_station_dispatch_snapshots",
            return_value=station_snapshots,
        ), patch(
            f"{module}.WorldModelTaskAssigner.select_robots",
            return_value={0: 7},
        ):
            choices = assigner._select_robots_interleaved(
                world,
                contexts,
                {
                    1: {"service": 0.2, "traffic": 0.2},
                    2: {"service": 0.2, "traffic": 0.2},
                },
            )

        self.assertEqual(choices, {0: 7})
        self.assertEqual(contexts, [healthy_context])
        self.assertEqual(assigner.stats["station_feedback_forced_defers"], 2)
        self.assertEqual(assigner.stats["station_feedback_selected_contexts"], 1)
        self.assertEqual(
            debt_feature_context_batches,
            [[blocked_context, blocked_context_2, healthy_context]],
        )
        self.assertEqual(dynamic_row_station_sets, [{2}])
        self.assertEqual(assigner.stats["dynamic_probe_steps"], 2)
        self.assertEqual(assigner.stats["station_context_defer_steps"], 2)
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_suppressed_station_ticks"
            ],
            1,
        )
        self.assertEqual(
            assigner.stats[
                "station_feedback_batch_filter_suppressed_context_ticks"
            ],
            2,
        )
        self.assertEqual(
            len(assigner._station_feedback_batch_filter_unique_contexts), 2
        )

    def test_off_subclass_matches_frozen_parent_decision_and_trace(self):
        def prepared(assigner):
            assigner.stats = defaultdict(int)
            assigner._station_context_defer_mass = {}
            assigner._station_context_defer_ticks = {}
            assigner._station_context_defer_bound = {}
            assigner._station_context_defer_clock = {}
            assigner._station_context_defer_last_update_tick = {}
            assigner._psi_last_final_budget = 1
            assigner._psi_distance = None
            assigner._psi_durations = None
            assigner._dynamic_reserved_robot_ids = set()
            assigner._prune_defer_mass = lambda world: None
            assigner._initial_pending_counts = lambda debt: {1: 1}
            assigner._sort_rows = lambda rows: list(rows)
            assigner._row_key = lambda row: (10, 20, 1)
            assigner._row_text = (
                lambda row: "order=10,pod=20,station=1"
            )
            assigner._virtual_idle_view = (
                lambda world, idle: nullcontext()
            )
            assigner.captured_trace = []
            assigner._append_dynamic_trace = (
                lambda record: assigner.captured_trace.append(record)
            )
            return assigner

        parent = prepared(
            StationContextDeferPipelinePhiDynamicPsiAssigner.__new__(
                StationContextDeferPipelinePhiDynamicPsiAssigner
            )
        )
        off = prepared(self._assigner(STATION_FEEDBACK_MODE_OFF))
        off._station_feedback_last_prepared_tick = None

        context = SimpleNamespace(order_id=10, pod_id=20, station_id=1)
        row = {
            "original_index": 0,
            "order_id": 10,
            "pod_id": 20,
            "station_id": 1,
            "context": context,
            "service": 0.2,
            "traffic": 0.2,
            "service_debt": 0.9,
            "backlog_score": 0.9,
            "age_score": 0.9,
            "free_flow_time": 5.0,
            "j_score": -0.7,
        }
        parent._dynamic_rows = lambda *args, **kwargs: [dict(row)]
        off._dynamic_rows = lambda *args, **kwargs: [dict(row)]
        world = SimpleNamespace(
            tick=1,
            get_idle_agents=lambda: [SimpleNamespace(agent_id=7)],
        )
        station_snapshot = StationDispatchSnapshot(
            station_id=1,
            capacity=3,
            physical=0,
            committed_in_transit=0,
            pre_pick=0,
            ready_unadmitted=0,
        )
        parent_contexts = [context]
        off_contexts = [context]
        module = (
            "Policies.TaskAssigner.WorldModelTaskAssigner."
            "station_context_defer_assigner"
        )
        with patch(
            f"{module}.build_context_debt_features", return_value={}
        ), patch(
            f"{module}.build_station_dispatch_snapshots",
            return_value={1: station_snapshot},
        ), patch(
            f"{module}.WorldModelTaskAssigner.select_robots",
            return_value={0: 7},
        ):
            parent_choices = parent._select_robots_interleaved(
                world, parent_contexts, {1: {"service": 0.2, "traffic": 0.2}}
            )
            off_choices = off._select_robots_interleaved(
                world, off_contexts, {1: {"service": 0.2, "traffic": 0.2}}
            )

        self.assertEqual(off_choices, parent_choices)
        self.assertEqual(off_contexts, parent_contexts)
        self.assertEqual(off.captured_trace, parent.captured_trace)
        parent_stats = dict(parent.stats)
        off_parent_stats = {
            key: value
            for key, value in off.stats.items()
            if not key.startswith("station_feedback_")
        }
        self.assertEqual(off_parent_stats, parent_stats)


if __name__ == "__main__":
    unittest.main()
