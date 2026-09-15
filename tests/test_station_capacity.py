from __future__ import annotations

from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_PHYSICAL_ONLY,
    SlotType,
    StationQueueState,
    StationSlot,
)


def _queue() -> StationQueueState:
    return StationQueueState(
        station_id=1,
        service=StationSlot((0, 0), SlotType.SERVICE, 0),
        queue_slots=[
            StationSlot((0, 1), SlotType.QUEUE, 0),
            StationSlot((0, 2), SlotType.QUEUE, 1),
        ],
        buffer_slots=[StationSlot((1, 2), SlotType.BUFFER, 0)],
        entry_position=(1, 1),
        exit_position=(1, 0),
    )


def test_physical_capacity_is_service_plus_queue_plus_buffer() -> None:
    queue = _queue()
    assert queue.capacity == 4
    queue.service.agent_id = 10
    queue.queue_slots[0].agent_id = 11
    assert queue.occupancy() == 2
    assert queue.physical_agent_ids() == {10, 11}


def test_committed_mode_rejects_the_fifth_unique_robot() -> None:
    queue = _queue()
    queue.set_admission_mode(STATION_ADMISSION_COMMITTED_V1)
    assert [queue.reserve(agent_id) for agent_id in (1, 2, 3, 4)] == [True] * 4
    assert queue.reserve(4) is True
    assert queue.reserve(5) is False
    audit = queue.admission_metrics()
    assert audit["capacity"] == 4
    assert audit["committed_load"] == 4
    assert audit["invariant_holds"] is True


def test_legacy_physical_only_counts_occupancy_for_admission() -> None:
    queue = _queue()
    assert queue.admission_mode == STATION_ADMISSION_PHYSICAL_ONLY
    assert [queue.reserve(agent_id) for agent_id in range(10, 16)] == [True] * 6
    assert queue.occupancy() == 0
    assert queue.committed_load() == 6
