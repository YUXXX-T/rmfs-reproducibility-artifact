import types
import unittest

from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
    STATION_ADMISSION_PHYSICAL_ONLY,
    SlotType,
    StationQueueState,
    StationSlot,
)


def _queue() -> StationQueueState:
    return StationQueueState(
        station_id=2,
        service=StationSlot((0, 0), SlotType.SERVICE, 0),
        queue_slots=[
            StationSlot((1, 0), SlotType.QUEUE, 0),
            StationSlot((2, 0), SlotType.QUEUE, 1),
        ],
        buffer_slots=[],
        entry_position=(3, 0),
        exit_position=(0, 1),
    )


class StationCommittedAdmissionTests(unittest.TestCase):
    def test_committed_mode_counts_in_transit_agents_against_capacity(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)

        self.assertTrue(queue.reserve(10))
        self.assertTrue(queue.reserve(11))
        self.assertTrue(queue.reserve(12))
        self.assertEqual(queue.occupancy(), 0)
        self.assertEqual(queue.committed_load(), queue.capacity)
        self.assertFalse(queue.reserve(13))
        self.assertEqual(queue.committed_agent_ids(), {10, 11, 12})
        self.assertTrue(queue.admission_metrics()["invariant_holds"])

    def test_repeated_reserve_is_idempotent_even_when_station_is_full(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)
        for agent_id in (10, 11, 12):
            self.assertTrue(queue.reserve(agent_id))

        self.assertTrue(queue.reserve(11))
        self.assertEqual(queue.committed_load(), 3)
        metrics = queue.admission_metrics()
        self.assertEqual(metrics["granted"], 3)
        self.assertEqual(metrics["idempotent"], 1)

    def test_physical_and_token_ledgers_are_unioned_without_double_counting(self):
        queue = _queue()
        queue.service.agent_id = 10
        queue._assigned_agents.update({10, 11})
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)

        self.assertEqual(queue.occupancy(), 1)
        self.assertEqual(queue.committed_agent_ids(), {10, 11})
        self.assertEqual(queue.committed_load(), 2)
        self.assertTrue(queue.reserve(12))
        self.assertFalse(queue.reserve(13))

    def test_unreserve_rolls_back_in_transit_but_not_physical_occupant(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)
        queue.service.agent_id = 10
        queue._assigned_agents.add(10)
        self.assertTrue(queue.reserve(11))

        queue.unreserve(11)
        self.assertNotIn(11, queue.committed_agent_ids())

        queue.unreserve(10)
        self.assertIn(10, queue.committed_agent_ids())
        self.assertIn(10, queue._assigned_agents)

    def test_successful_station_exit_releases_slot_and_token(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)
        queue.service.agent_id = 10
        queue._assigned_agents.add(10)
        agent = types.SimpleNamespace(
            agent_id=10,
            position=queue.service.position,
            carried_pod_id=None,
        )
        world = types.SimpleNamespace(
            agents=[agent],
            get_agent=lambda agent_id: agent if agent_id == 10 else None,
        )

        self.assertTrue(queue.release_to_exit(10, world))
        self.assertEqual(agent.position, queue.exit_position)
        self.assertEqual(queue.occupancy(), 0)
        self.assertEqual(queue.committed_load(), 0)

    def test_legacy_mode_remains_default_for_frozen_experiments(self):
        queue = _queue()
        self.assertEqual(queue.admission_mode, STATION_ADMISSION_PHYSICAL_ONLY)

        for agent_id in (10, 11, 12, 13):
            self.assertTrue(queue.reserve(agent_id))
        self.assertEqual(queue.occupancy(), 0)
        self.assertEqual(queue.committed_load(), 4)
        self.assertEqual(
            queue.admission_metrics()["over_capacity_grants"], 1
        )
        with self.assertRaises(RuntimeError):
            queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)

    def test_all_modes_expose_transaction_rollback_instrumentation(self):
        queue = _queue()
        self.assertTrue(queue.reserve(10))
        queue.unreserve(10, reason="entry_occupied")
        self.assertTrue(queue.reserve(11))
        queue.unreserve(11, reason="path_failure")
        self.assertTrue(queue.reserve(12))
        queue.unreserve(12, reason="unexpected")

        self.assertEqual(queue.committed_load(), 0)
        self.assertEqual(
            queue.admission_metrics()["rollbacks"],
            {"entry_occupied": 1, "path_failure": 1, "other": 1},
        )

    def test_dynamic_eta_mode_overbooks_healthy_station_up_to_two_capacity(self):
        queue = _queue()
        queue.configure_dynamic_admission(service_ticks=5)
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)

        for agent_id, eta in ((10, 2), (11, 6), (12, 14), (13, 24), (14, 24), (15, 24)):
            self.assertTrue(
                queue.reserve(agent_id, request_tick=0, eta_ticks=eta)
            )
        self.assertEqual(queue.capacity, 3)
        self.assertEqual(queue.committed_load(), 6)
        self.assertEqual(queue.dynamic_hard_limit(), 6)
        self.assertFalse(queue.reserve(16, request_tick=0, eta_ticks=30))

        metrics = queue.admission_metrics()
        self.assertTrue(metrics["invariant_holds"])
        self.assertEqual(metrics["dynamic_eta"]["overbooking_grants"], 3)
        self.assertEqual(metrics["dynamic_eta"]["rejected_hard_limit"], 1)

    def test_dynamic_eta_weighting_rejects_near_burst_before_hard_limit(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)

        for agent_id in range(10, 16):
            accepted = queue.reserve(agent_id, request_tick=0, eta_ticks=2)
            if agent_id < 16:
                self.assertTrue(accepted)
        self.assertEqual(queue.committed_load(), 6)

        # A tighter healthy weighted budget rejects a new near-term arrival
        # even though the count hard limit has been raised independently.
        queue2 = _queue()
        queue2.configure_dynamic_admission(
            max_committed_multiplier=3.0,
            healthy_extra_ratio=0.5,
        )
        queue2.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        for agent_id in (20, 21, 22, 23):
            self.assertTrue(queue2.reserve(agent_id, request_tick=0, eta_ticks=2))
        self.assertFalse(queue2.reserve(24, request_tick=0, eta_ticks=2))
        self.assertEqual(
            queue2.admission_metrics()["dynamic_eta"]["rejected_weighted_limit"],
            1,
        )

    def test_dynamic_health_brake_only_blocks_new_tokens(self):
        queue = _queue()
        queue.configure_dynamic_admission(
            caution_block_streak=2,
            brake_block_streak=3,
            caution_release_drought=20,
            brake_release_drought=30,
        )
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        self.assertTrue(queue.reserve(10, request_tick=0, eta_ticks=20))
        self.assertTrue(queue.reserve(11, request_tick=0, eta_ticks=20))
        self.assertTrue(queue.reserve(12, request_tick=0, eta_ticks=20))

        blocker = types.SimpleNamespace(agent_id=99, position=queue.entry_position)
        world = types.SimpleNamespace(
            tick=0,
            agents=[blocker],
            task_state=types.SimpleNamespace(
                get_active_task_for_agent=lambda agent_id: None
            ),
        )
        for tick in range(3):
            world.tick = tick
            queue.observe_tick(world)

        self.assertEqual(queue.dynamic_health_state(), "brake")
        self.assertEqual(queue.committed_agent_ids(), {10, 11, 12})
        self.assertFalse(queue.reserve(13, request_tick=3, eta_ticks=30))
        self.assertEqual(queue.committed_agent_ids(), {10, 11, 12})

    def test_dynamic_full_station_with_regular_release_is_not_braked(self):
        queue = _queue()
        queue.configure_dynamic_admission(
            caution_release_drought=2,
            brake_release_drought=3,
            caution_block_streak=20,
            brake_block_streak=30,
        )
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        for slot, agent_id in zip(queue._all_slots(), (10, 11, 12)):
            slot.agent_id = agent_id
            queue._assigned_agents.add(agent_id)
        world = types.SimpleNamespace(
            tick=0,
            agents=[],
            task_state=types.SimpleNamespace(
                get_active_task_for_agent=lambda agent_id: None
            ),
        )
        for tick in range(3):
            world.tick = tick
            queue._dynamic_last_release_tick = tick
            queue.observe_tick(world)
        self.assertEqual(queue.dynamic_health_state(), "healthy")

    def test_dynamic_eta_arrival_and_release_are_observed(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        self.assertTrue(queue.reserve(10, request_tick=0, eta_ticks=4))
        agent = types.SimpleNamespace(
            agent_id=10,
            position=queue.entry_position,
            carried_pod_id=None,
            status=None,
        )
        world = types.SimpleNamespace(
            tick=6,
            agents=[agent],
            get_agent=lambda agent_id: agent if agent_id == 10 else None,
            pod_state=types.SimpleNamespace(get_pod=lambda pod_id: None),
        )

        self.assertTrue(queue.check_in_from_entry(10, world))
        dynamic = queue.admission_metrics()["dynamic_eta"]
        self.assertEqual(dynamic["eta_sample_count"], 1)
        self.assertEqual(dynamic["eta_signed_error_mean"], 2.0)

        slot = queue.queue_slots[-1]
        slot.agent_id = 10
        agent.position = slot.position
        world.tick = 10
        self.assertTrue(queue.release_to_exit(10, world))
        dynamic = queue.admission_metrics()["dynamic_eta"]
        self.assertEqual(dynamic["release_count"], 1)
        self.assertEqual(queue.committed_load(), 0)

    def test_dynamic_legacy_check_in_repairs_token_until_exit(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        agent = types.SimpleNamespace(
            agent_id=10,
            position=queue.entry_position,
            carried_pod_id=None,
            status=None,
        )
        world = types.SimpleNamespace(
            tick=4,
            agents=[agent],
            get_agent=lambda agent_id: agent if agent_id == 10 else None,
            pod_state=types.SimpleNamespace(get_pod=lambda pod_id: None),
        )

        self.assertTrue(queue.check_in_from_entry(10, world))
        self.assertIn(10, queue._assigned_agents)
        self.assertEqual(queue.committed_agent_ids(), {10})

        slot = queue.queue_slots[-1]
        slot.agent_id = 10
        agent.position = slot.position
        queue.unreserve(10)
        self.assertIn(10, queue.committed_agent_ids())

    def test_dynamic_transaction_rollback_reason_is_counted(self):
        queue = _queue()
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        self.assertTrue(queue.reserve(10, request_tick=0, eta_ticks=12))
        queue.unreserve(10, reason="path_failure")

        dynamic = queue.admission_metrics()["dynamic_eta"]
        self.assertEqual(dynamic["rollbacks"]["path_failure"], 1)
        self.assertEqual(queue.committed_load(), 0)

    def test_dynamic_eta_windows_limit_synchronized_near_arrivals(self):
        queue = _queue()
        queue.configure_dynamic_admission(
            max_committed_multiplier=3.0,
            healthy_extra_ratio=2.0,
        )
        queue.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)

        # C=3, cold-start release cadence=7 and healthy slack=2, so the near
        # window admits six due commitments (3 slots + 1 release + 2 slack).
        for agent_id in range(10, 16):
            self.assertTrue(
                queue.reserve(agent_id, request_tick=0, eta_ticks=8)
            )
        self.assertFalse(queue.reserve(16, request_tick=0, eta_ticks=8))
        dynamic = queue.admission_metrics()["dynamic_eta"]
        self.assertEqual(dynamic["rejected_near_window"], 1)

        # A far commitment remains eligible because it does not join the near
        # or mid arrival burst and the absolute hard limit is still open.
        self.assertTrue(queue.reserve(17, request_tick=0, eta_ticks=30))


if __name__ == "__main__":
    unittest.main()
