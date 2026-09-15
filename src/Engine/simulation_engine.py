"""
Simulation Engine Module
========================
Core simulation loop orchestrating order generation, task assignment,
path planning, agent movement, and pickup/delivery logic.
"""

import signal
import sys
from typing import Optional

from Config.config_loader import SimulationConfig
from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.order_state import OrderStatus
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2
from Policies.OrderGenerator import BaseOrderGenerator
from Policies.TaskAssigner import BaseTaskAssigner
from Policies.TaskAssigner.base_task_assigner import (
    fulfilled_pod_ids_for_order,
)
from Policies.PathPlanner import BasePathPlanner
from Debug.logger import SimLogger
from Metrics.tracker import MetricsTracker


class SimulationEngine:
    """
    Main simulation engine running a continuous tick loop.

    The loop runs indefinitely until interrupted (Ctrl+C), at which point
    it prints a summary and exits gracefully.

    参数
    ----------
    config : SimulationConfig
        Loaded simulation configuration.
    order_generator : BaseOrderGenerator
        Policy for generating orders.
    task_assigner : BaseTaskAssigner
        Policy for assigning tasks to agents.
    path_planner : BasePathPlanner
        Policy for computing agent paths.
    visualizer : object or None
        Optional visualizer with a `render(world_state)` method.
    """

    def __init__(
        self,
        config: SimulationConfig,
        order_generator: BaseOrderGenerator,
        task_assigner: BaseTaskAssigner,
        path_planner: BasePathPlanner,
        visualizer=None,
    ):
        self.config = config
        self.world = WorldState(config)
        self.order_generator = order_generator
        self.task_assigner = task_assigner
        self.path_planner = path_planner
        self.visualizer = visualizer

        self.logger = SimLogger(
            "Engine",
            level=config.simulation.log_level,
            log_file=config.simulation.log_file,
        )
        self.metrics = MetricsTracker()
        self.on_tick_callbacks: list = []
        self.pre_assignment_callbacks: list = []
        self._running = True
        self._station_dispatch_sequence = 0

        # Conflict details from the most recent tick — populated by
        # _detect_conflicts so callbacks (e.g. SnapshotCollector) can serialize them.
        self.last_vertex_conflicts: list[tuple] = []   # [((r,c), [agent_ids...]), ...]
        self.last_swap_conflicts: list[tuple] = []     # [((r1,c1), (r2,c2), aid_a, aid_b), ...]
        # Read-only cumulative diagnostics used by planner-adaptation audits.
        # Existing MetricsTracker definitions remain unchanged.
        self.total_vertex_conflicts_detected = 0
        self.total_swap_conflicts_detected = 0

    def _seed_initial_orders(self):
        """Pre-fill the order pool before tick 0 if configured."""
        target = self.config.simulation.initial_order_pool_size
        if target <= 0:
            return
        count = 0
        while len(self.world.order_state.get_pending_orders()) < target:
            if hasattr(self.order_generator, "generate_one"):
                order = self.order_generator.generate_one(self.world, created_at=0)
                if order is None:
                    break
                self.world.order_state.add_order(order)
                count += 1
            else:
                break
        if count > 0:
            self.logger.info(f"Seeded {count} initial orders (target={target})")

    def _refill_order_backlog(self, tick: int):
        """Ensure pending orders stay above backlog_floor if configured."""
        floor = self.config.simulation.backlog_floor
        if floor <= 0:
            return
        count = 0
        pending = len(self.world.order_state.get_pending_orders())
        while pending < floor:
            if hasattr(self.order_generator, "generate_one"):
                order = self.order_generator.generate_one(self.world, created_at=tick)
                if order is None:
                    break
                self.world.order_state.add_order(order)
                pending += 1
                count += 1
            else:
                break
        if count > 0:
            self.logger.debug(f"[Tick {tick}] Backlog refill: +{count} orders (floor={floor})")

    def run(self):
        """
        Start the continuous simulation loop.

        Press Ctrl+C to stop gracefully.
        """
        # 注册信号处理器以优雅停机（SIGINT=Ctrl+C, SIGTERM=`timeout` / `kill`）
        original_int = signal.getsignal(signal.SIGINT)
        original_term = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            self.logger.info(f"{sig_name} received. Finishing current tick...")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        self.logger.info("=" * 60)
        self.logger.info("MAS-RMFS Simulation Started")
        self.logger.info(f"  Map: {self.world.map_state.rows}x{self.world.map_state.cols}")
        self.logger.info(f"  Agents: {len(self.world.agents)}")
        self.logger.info(f"  Pods: {self.world.pod_state.total_pods}")
        self.logger.info(f"  Stations: {len(self.world.map_state.station_positions)}")
        self.logger.info("  Press Ctrl+C to stop.")
        self.logger.info("=" * 60)

        self._seed_initial_orders()

        try:
            while self._running:
                self._tick()
                max_ticks = self.config.simulation.max_ticks
                if max_ticks and self.world.tick >= max_ticks:
                    self.logger.info(
                        f"Reached max_ticks={max_ticks}; stopping."
                    )
                    self._running = False
                    break
                if self.config.simulation.tick_delay > 0:
                    import time
                    time.sleep(self.config.simulation.tick_delay)
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self._print_summary()
            signal.signal(signal.SIGINT, original_int)
            signal.signal(signal.SIGTERM, original_term)

    def _tick(self):
        """Execute one simulation tick."""
        tick = self.world.tick

        # --- Save tick-start positions for stationary_ticks update ---
        tick_start_positions = {a.agent_id: a.position for a in self.world.agents}

        # --- 步骤 1：生成订单 ---
        new_orders = self.order_generator.generate(self.world)
        for order in new_orders:
            self.world.order_state.add_order(order)
            self.logger.info(
                f"[Tick {tick}] New Order #{order.order_id}: "
                f"sku_demands={order.sku_demands} -> station {order.station_id}"
            )

        # --- 步骤 1.25：backlog refill (确保 pending 充足) ---
        mode = self.config.simulation.backlog_refill_mode
        if mode == "on_pre_assignment":
            self._refill_order_backlog(tick)

        # A conditional World-Model robot policy receives a fixed order/pod
        # context. Materialise that inventory context outside the assigner so
        # no PodRetriever or Greedy logic enters the robot decision call.
        if getattr(
            self.task_assigner, "requires_external_fixed_context", False
        ):
            from Policies.TaskAssigner.context_assignment import (
                materialize_fixed_order_pod_contexts,
            )
            materialize_fixed_order_pod_contexts(
                self.world,
                getattr(self.task_assigner, "pod_retriever", None),
            )

        # --- 步骤 1.5：pre-assignment 回调（世界模型数据采集） ---
        for cb in self.pre_assignment_callbacks:
            cb(self)

        # --- 步骤 2：分配任务（计时） ---
        self.metrics.start_timer("assign")
        new_tasks = self.task_assigner.assign(self.world)
        assign_ms = self.metrics.stop_timer("assign")
        if self._uses_dispatch_priority_station_admission():
            new_tasks = self._stamp_station_dispatch_sequences(new_tasks)
        for task in new_tasks:
            if task.created_at is None:
                task.created_at = tick
            if task.agent_id is not None and task.assigned_at is None:
                task.assigned_at = tick
            if task.free_flow_time is None:
                task.free_flow_time = self._estimate_free_flow(task)
            self.logger.debug(
                f"[Tick {tick}] Task #{task.task_id} ({task.task_type.name}) "
                f"assigned to Agent #{task.agent_id}"
            )

        # --- Step 3: Station queue cascade ---
        self.world.station_state.tick(self.world)

        # --- Step 3b: Process station exits (EXITING -> exit_position) ---
        self._process_station_exits(tick)

        # --- Step 3c: Absorb agents stranded at entry from previous tick ---
        absorbed_entries = set()
        self._check_queue_arrivals(tick, absorbed_entries)

        # --- Step 3d: Clear stale DELIVER paths near free entry ---
        self._clear_stale_entry_paths(tick)

        # --- Step 4: Plan paths & activate tasks（计时） ---
        self.metrics.start_timer("plan")
        self._plan_and_activate(tick)
        plan_ms = self.metrics.stop_timer("plan")

        # --- Step 5: Capture pre-move positions & Move agents ---
        prev_positions = {agent.agent_id: agent.position for agent in self.world.agents}
        self._move_agents(tick)

        # --- Step 5b: Check queue arrivals (new arrivals) ---
        self._check_queue_arrivals(tick, absorbed_entries)

        # --- 步骤 6：检测冲突 (vertex & oncoming/swap) ---
        vertex_conflicts = self._detect_conflicts(tick, prev_positions)

        # --- Step 7: Handle pickups, deliveries, returns ---
        self._handle_actions(tick)

        # --- 步骤 7：检查订单完成 ---
        self._check_order_completion(tick)

        # --- 步骤 7.5：on_completed backlog refill ---
        mode = self.config.simulation.backlog_refill_mode
        if mode == "on_completed":
            self._refill_order_backlog(tick)

        # --- 步骤 7.6：Update stationary_ticks (before metrics/callbacks) ---
        from WorldState.risk import _update_stationary_ticks
        _update_stationary_ticks(self.world, tick_start_positions)

        # --- 步骤 8：记录指标 ---
        self.metrics.record(self.world, vertex_conflicts, assign_ms, plan_ms)

        # --- 步骤 8.5：tick 回调（快照采集等） ---
        for cb in self.on_tick_callbacks:
            cb(self)

        # --- 步骤 9：可视化（可选） ---
        if self.visualizer:
            self.visualizer.render(self.world)

        # Advance tick
        self.world.advance_tick()

    def _uses_fifo_station_admission(self) -> bool:
        return any(
            queue.admission_mode == STATION_ADMISSION_COMMITTED_FIFO_V2
            for queue in self.world.station_state.stations.values()
        )

    def _uses_explicit_station_waiting(self) -> bool:
        return any(
            queue.uses_explicit_waiting
            for queue in self.world.station_state.stations.values()
        )

    def _uses_dispatch_priority_station_admission(self) -> bool:
        return any(
            queue.uses_dispatch_priority_waiting
            for queue in self.world.station_state.stations.values()
        )

    def _allocate_station_dispatch_sequence(self) -> int:
        self._station_dispatch_sequence = int(
            getattr(self, "_station_dispatch_sequence", 0)
        ) + 1
        return self._station_dispatch_sequence

    def _stamp_station_dispatch_sequences(self, tasks) -> list:
        """Stamp DELIVER chains in the exact order returned by the assigner."""
        rows = list(tasks)
        existing_by_chain = {}
        for task in rows:
            if task.task_type != TaskType.DELIVER:
                continue
            sequence = getattr(task, "station_dispatch_sequence", None)
            if sequence is None:
                continue
            key = (task.order_id, task.pod_id, task.agent_id)
            existing_by_chain[key] = int(sequence)

        sequence_by_chain = {}
        for task in rows:
            if task.task_type != TaskType.DELIVER:
                continue
            key = (task.order_id, task.pod_id, task.agent_id)
            if key not in sequence_by_chain:
                sequence = existing_by_chain.get(key)
                if sequence is None:
                    sequence = self._allocate_station_dispatch_sequence()
                sequence_by_chain[key] = sequence
            task.station_dispatch_sequence = int(sequence_by_chain[key])
        return rows

    def _station_wait_sequence(self, agent) -> Optional[int]:
        """Return a waiting robot's active station queue sequence, if any."""
        if agent.status != AgentStatus.WAITING_ASSIGNED:
            return None
        station_id = agent.station_waiting_station_id
        if station_id is None:
            return None
        queue = self.world.station_state.get_queue(station_id)
        if queue is None:
            return None
        return queue.waiting_reservation_sequence(agent.agent_id)

    def _mark_station_waiting(self, agent, queue, task, tick: int) -> None:
        """Enter the explicit post-PICK waiting-assigned phase."""
        if not queue.uses_explicit_waiting:
            return
        sequence = queue.waiting_reservation_sequence(agent.agent_id)
        if sequence is None:
            sequence = queue.enqueue_waiting_reservation(
                agent.agent_id,
                request_tick=tick,
                priority_sequence=getattr(
                    task, "station_dispatch_sequence", None
                ),
            )
        agent.clear_path()
        agent.wait_ticks = 0
        agent.assigned_task_id = task.task_id
        agent.mark_station_waiting(
            station_id=task.station_id,
            since_tick=(
                agent.station_waiting_since_tick
                if agent.station_waiting_since_tick is not None
                else tick
            ),
            sequence=sequence,
        )

    def _prune_station_waiting(self, tick: int) -> None:
        """Remove stale station requests and recover their agent state."""
        station_state = self.world.station_state
        if not self._uses_explicit_station_waiting():
            return
        for station_id, queue in station_state.stations.items():
            valid = set()
            for agent in self.world.agents:
                if agent.status != AgentStatus.WAITING_ASSIGNED:
                    continue
                next_task = self.world.task_state.get_next_task_for_agent(
                    agent.agent_id
                )
                if (
                    next_task is not None
                    and next_task.task_type == TaskType.DELIVER
                    and next_task.station_id == station_id
                ):
                    valid.add(agent.agent_id)
            removed = queue.prune_waiting_reservations(valid)
            for agent_id in removed:
                agent = self.world.get_agent(agent_id)
                if agent is None:
                    continue
                agent.clear_station_waiting()
                agent.assigned_task_id = None
                if agent.carried_pod_id is None:
                    agent.status = AgentStatus.IDLE
                else:
                    # A stale reservation must not make a carried pod idle;
                    # the next assigned chain, if any, will be activated by the
                    # normal generic branch.
                    agent.status = AgentStatus.CARRYING

    def _plan_and_activate(self, tick: int):
        """Plan paths for agents that have assigned tasks but no active path."""
        station_state = self.world.station_state

        self._prune_station_waiting(tick)

        # Build handoff blocked set for active stations
        active_station_ids = set()
        for agent in self.world.agents:
            if agent.status == AgentStatus.EXITING:
                t = self.world.task_state.get_active_task_for_agent(agent.agent_id)
                if t:
                    active_station_ids.add(t.station_id)
            elif agent.status == AgentStatus.CARRYING and not agent.has_path:
                t = self.world.task_state.get_active_task_for_agent(agent.agent_id)
                if t and t.task_type == TaskType.DELIVER:
                    active_station_ids.add(t.station_id)

        handoff_blocked = set()
        for sq in station_state.stations.values():
            if sq.station_id not in active_station_ids:
                continue
            if sq.entry_position:
                handoff_blocked.add(sq.entry_position)
            if sq.exit_position:
                handoff_blocked.add(sq.exit_position)

        def sort_key(agent):
            waiting = agent.status == AgentStatus.WAITING_ASSIGNED
            sequence = self._station_wait_sequence(agent)
            if waiting:
                # The active waiting sequence is primary for admission
                # waiters. FIFO uses enqueue order; the new ETA queue uses the
                # scheduler's original dispatch order.
                return (
                    0,
                    sequence if sequence is not None else 10**12,
                    agent.agent_id,
                )
            return (
                1,
                0 if agent.position in handoff_blocked else 1,
                10**12,
                agent.agent_id,
            )

        agents_sorted = sorted(self.world.agents, key=sort_key)

        # Single-step joint planners such as PIBT must see the complete set of
        # robots that the engine will actually move in this tick.  Keeping this
        # behind an explicit capability flag leaves the historical sequential
        # PP/A*/external-solver path completely unchanged.
        if (
            getattr(self.path_planner, "supports_batch_planning", False)
            and callable(getattr(self.path_planner, "plan_batch", None))
        ):
            self._plan_and_activate_batch(
                tick,
                station_state,
                handoff_blocked,
                agents_sorted,
            )
            return

        # --- Phase 1: activate tasks, with DELIVER special handling ---
        agents_need_plan = []
        for agent in agents_sorted:
            if agent.status in (AgentStatus.QUEUING, AgentStatus.DELIVERING,
                                AgentStatus.EXITING):
                continue
            if agent.is_idle and agent.position in handoff_blocked:
                self.logger.warning(
                    f"[Tick {tick}] Agent #{agent.agent_id} is IDLE on handoff "
                    f"cell {agent.position} — may block station exit/entry"
                )
            if agent.is_idle or (
                agent.is_waiting
                and agent.status != AgentStatus.WAITING_ASSIGNED
            ):
                continue

            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)

            # -- DELIVER branch: fully transactional, separate from generic flow --
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is not None and next_task.task_type == TaskType.DELIVER:
                    queue = station_state.get_queue(next_task.station_id)
                    if queue is None or queue.entry_position is None:
                        continue

                    if agent.status == AgentStatus.WAITING_ASSIGNED:
                        # FIFO reconsiders only its head. The dispatch-priority
                        # ETA queue attempts ready waiters in scheduler order,
                        # allowing a later ETA-feasible request to use capacity
                        # when an earlier request is currently infeasible.
                        if not queue.can_promote_waiting_reservation(
                            agent.agent_id
                        ):
                            continue

                    er, ec = queue.entry_position
                    ar, ac = agent.position
                    rough_eta = abs(er - ar) + abs(ec - ac)
                    if not queue.reserve(
                        agent.agent_id,
                        request_tick=tick,
                        eta_ticks=rough_eta,
                    ):
                        if queue.uses_explicit_waiting:
                            self._mark_station_waiting(
                                agent, queue, next_task, tick
                            )
                        continue

                    goal = queue.entry_position

                    # Don't send another agent when entry is occupied
                    if any(a.position == goal and a.agent_id != agent.agent_id
                           and not a.has_path
                           for a in self.world.agents):
                        queue.unreserve(
                            agent.agent_id, reason="entry_occupied"
                        )
                        if queue.uses_explicit_waiting:
                            queue.requeue_waiting_reservation(
                                agent.agent_id,
                                request_tick=tick,
                                priority_sequence=getattr(
                                    next_task,
                                    "station_dispatch_sequence",
                                    None,
                                ),
                            )
                            self._mark_station_waiting(
                                agent, queue, next_task, tick
                            )
                        continue
                    extra_blocked = handoff_blocked - {goal, agent.position}
                    # A zero-length path is a successful activation when the
                    # robot is already on the station entry.  For any other
                    # position, an empty result still means planning failed.
                    path = (
                        []
                        if agent.position == goal
                        else self.path_planner.plan(
                            agent, goal, self.world,
                            extra_blocked=extra_blocked,
                        )
                    )
                    if not path and agent.position != goal:
                        queue.unreserve(
                            agent.agent_id, reason="path_failure"
                        )
                        if queue.uses_explicit_waiting:
                            queue.requeue_waiting_reservation(
                                agent.agent_id,
                                request_tick=tick,
                                priority_sequence=getattr(
                                    next_task,
                                    "station_dispatch_sequence",
                                    None,
                                ),
                            )
                            self._mark_station_waiting(
                                agent, queue, next_task, tick
                            )
                        self.logger.warning(
                            f"[Tick {tick}] Agent #{agent.agent_id} could not find "
                            f"path to entry {goal} for DELIVER"
                        )
                        continue

                    next_task.status = TaskStatus.IN_PROGRESS
                    if next_task.started_at is None:
                        next_task.started_at = tick
                    agent.status = AgentStatus.CARRYING
                    agent.assigned_task_id = next_task.task_id
                    if queue.uses_explicit_waiting:
                        agent.clear_station_waiting()
                    agent.assign_path(path)
                    agent.plan_failed_streak = 0
                    self._record_path_len(agent, path)
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} DELIVER → "
                        f"entry {goal} ({len(path)} steps)"
                    )
                    continue

            # -- Generic branch: PICK, RETURN, already-active tasks --
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is None:
                    continue
                next_task.status = TaskStatus.IN_PROGRESS
                if next_task.started_at is None:
                    next_task.started_at = tick
                active_task = next_task

                if active_task.task_type == TaskType.PICK:
                    agent.status = AgentStatus.MOVING_TO_POD
                elif active_task.task_type == TaskType.RETURN:
                    agent.status = AgentStatus.RETURNING

                agent.assigned_task_id = active_task.task_id

            if not agent.has_path:
                if (active_task.task_type == TaskType.DELIVER
                        and agent.status == AgentStatus.CARRYING):
                    queue = station_state.get_queue(active_task.station_id)
                    if queue and queue.entry_position:
                        goal = queue.entry_position
                    else:
                        goal = active_task.destination
                else:
                    goal = active_task.destination
                if agent.position == goal:
                    agent.plan_failed_streak = 0
                    continue
                extra_blocked = handoff_blocked - {goal, agent.position}
                path = self.path_planner.plan(
                    agent, goal, self.world,
                    extra_blocked=extra_blocked if extra_blocked else None,
                )
                if path:
                    agent.assign_path(path)
                    agent.plan_failed_streak = 0
                    self._record_path_len(agent, path)
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} planned path "
                        f"to {goal} ({len(path)} steps)"
                    )
                else:
                    agent.plan_failed_streak += 1
                    self.logger.warning(
                        f"[Tick {tick}] Agent #{agent.agent_id} could not find "
                        f"path to {goal}"
                    )
                    # Nudge disabled: assigns paths outside the PP
                    # reservation table, causing vertex conflicts.
                    # if agent.position in handoff_blocked:
                    #     self._nudge_idle_neighbors(agent, tick)

    def _plan_and_activate_batch(
        self,
        tick: int,
        station_state,
        handoff_blocked: set,
        agents_sorted: list,
    ) -> None:
        """RMFS activation path for one-step joint planners (currently PIBT).

        Station admission and task lifecycle decisions are performed in the
        same deterministic order as the sequential path.  Only after those
        decisions are frozen is the exact movable robot set sent to the joint
        planner.  Robots omitted from the request are therefore physically
        pinned by the planner for this tick.
        """
        requests = []

        def finish_deliver_activation(request, path) -> None:
            agent = request["agent"]
            task = request["task"]
            queue = request["queue"]
            goal = request["goal"]
            task.status = TaskStatus.IN_PROGRESS
            if task.started_at is None:
                task.started_at = tick
            agent.status = AgentStatus.CARRYING
            agent.assigned_task_id = task.task_id
            if queue.uses_explicit_waiting:
                agent.clear_station_waiting()
            agent.assign_path(path)
            agent.plan_failed_streak = 0
            self._record_path_len(agent, path)
            self.logger.debug(
                f"[Tick {tick}] Agent #{agent.agent_id} DELIVER -> "
                f"entry {goal} ({len(path)} steps)"
            )

        for agent in agents_sorted:
            if agent.status in (
                AgentStatus.QUEUING,
                AgentStatus.DELIVERING,
                AgentStatus.EXITING,
            ):
                continue
            if agent.is_idle and agent.position in handoff_blocked:
                self.logger.warning(
                    f"[Tick {tick}] Agent #{agent.agent_id} is IDLE on handoff "
                    f"cell {agent.position} - may block station exit/entry"
                )
            if agent.is_idle or (
                agent.is_waiting
                and agent.status != AgentStatus.WAITING_ASSIGNED
            ):
                continue

            active_task = self.world.task_state.get_active_task_for_agent(
                agent.agent_id
            )

            # DELIVER activation remains transactional with station admission.
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(
                    agent.agent_id
                )
                if (
                    next_task is not None
                    and next_task.task_type == TaskType.DELIVER
                ):
                    queue = station_state.get_queue(next_task.station_id)
                    if queue is None or queue.entry_position is None:
                        continue

                    if agent.status == AgentStatus.WAITING_ASSIGNED:
                        if not queue.can_promote_waiting_reservation(
                            agent.agent_id
                        ):
                            continue

                    er, ec = queue.entry_position
                    ar, ac = agent.position
                    rough_eta = abs(er - ar) + abs(ec - ac)
                    if not queue.reserve(
                        agent.agent_id,
                        request_tick=tick,
                        eta_ticks=rough_eta,
                    ):
                        if queue.uses_explicit_waiting:
                            self._mark_station_waiting(
                                agent, queue, next_task, tick
                            )
                        continue

                    goal = queue.entry_position
                    if any(
                        other.position == goal
                        and other.agent_id != agent.agent_id
                        and not other.has_path
                        for other in self.world.agents
                    ):
                        queue.unreserve(
                            agent.agent_id, reason="entry_occupied"
                        )
                        if queue.uses_explicit_waiting:
                            queue.requeue_waiting_reservation(
                                agent.agent_id,
                                request_tick=tick,
                                priority_sequence=getattr(
                                    next_task,
                                    "station_dispatch_sequence",
                                    None,
                                ),
                            )
                            self._mark_station_waiting(
                                agent, queue, next_task, tick
                            )
                        continue

                    request = {
                        "kind": "deliver",
                        "agent": agent,
                        "task": next_task,
                        "queue": queue,
                        "goal": goal,
                    }
                    if agent.position == goal:
                        finish_deliver_activation(request, [])
                    else:
                        requests.append(request)
                    continue

            # Generic PICK / RETURN / already-active task activation.
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(
                    agent.agent_id
                )
                if next_task is None:
                    continue
                next_task.status = TaskStatus.IN_PROGRESS
                if next_task.started_at is None:
                    next_task.started_at = tick
                active_task = next_task
                if active_task.task_type == TaskType.PICK:
                    agent.status = AgentStatus.MOVING_TO_POD
                elif active_task.task_type == TaskType.RETURN:
                    agent.status = AgentStatus.RETURNING
                agent.assigned_task_id = active_task.task_id

            if agent.has_path:
                continue
            if (
                active_task.task_type == TaskType.DELIVER
                and agent.status == AgentStatus.CARRYING
            ):
                queue = station_state.get_queue(active_task.station_id)
                if queue and queue.entry_position:
                    goal = queue.entry_position
                else:
                    goal = active_task.destination
            else:
                goal = active_task.destination

            if agent.position == goal:
                agent.plan_failed_streak = 0
                continue
            requests.append({
                "kind": "generic",
                "agent": agent,
                "task": active_task,
                "queue": None,
                "goal": goal,
            })

        if not requests:
            return

        agents_with_goals = [
            (request["agent"], request["goal"])
            for request in requests
        ]
        paths = self.path_planner.plan_batch(
            agents_with_goals,
            self.world,
            extra_blocked=handoff_blocked if handoff_blocked else None,
        )
        expected_ids = {
            int(request["agent"].agent_id) for request in requests
        }
        actual_ids = {int(agent_id) for agent_id in paths}
        if actual_ids != expected_ids:
            raise RuntimeError(
                "batch planner returned a different robot set: "
                f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
            )
        for request in requests:
            agent = request["agent"]
            path = paths[agent.agent_id]
            if not isinstance(path, list) or not path:
                raise RuntimeError(
                    "single-step batch planner must return a non-empty list "
                    f"for Agent #{agent.agent_id}; got {path!r}"
                )

        for request in requests:
            agent = request["agent"]
            path = paths[agent.agent_id]
            if request["kind"] == "deliver":
                finish_deliver_activation(request, path)
                continue
            agent.assign_path(path)
            agent.plan_failed_streak = 0
            self._record_path_len(agent, path)
            self.logger.debug(
                f"[Tick {tick}] Agent #{agent.agent_id} planned batch path "
                f"to {request['goal']} ({len(path)} steps)"
            )

    def _move_agents(self, tick: int):
        """Move each agent one step along their path.

        Iterative conflict resolution: compute intended moves, block
        conflicting ones, propagate until stable, then execute.
        """
        movable = []
        for agent in self.world.agents:
            agent.previous_position = agent.position
            agent.stuck_this_tick = False
            agent.traffic_blocked_this_tick = False
            agent.moved_this_tick = False
            if agent.status in (AgentStatus.QUEUING, AgentStatus.DELIVERING,
                                AgentStatus.EXITING):
                continue
            if agent.is_waiting:
                continue
            if agent.has_path:
                movable.append(agent)

        intended = {}
        for agent in movable:
            intended[agent.agent_id] = agent.path[agent.path_index]

        occupied = {
            a.position for a in self.world.agents
            if a.agent_id not in intended
        }

        blocked_aids: set = set()
        changed = True
        while changed:
            changed = False
            target_counts: dict = {}
            for aid, pos in intended.items():
                if aid in blocked_aids:
                    continue
                target_counts.setdefault(pos, []).append(aid)

            for pos, aids in target_counts.items():
                if len(aids) > 1 or pos in occupied:
                    for aid in aids:
                        if aid not in blocked_aids:
                            blocked_aids.add(aid)
                            ag = self.world.get_agent(aid)
                            occupied.add(ag.position)
                            changed = True

        for agent in movable:
            if agent.agent_id in blocked_aids:
                agent.clear_path()
                agent.stuck_this_tick = True
                agent.traffic_blocked_this_tick = True
                continue
            old_pos = agent.position
            new_pos = agent.advance()
            if new_pos == old_pos:
                agent.stuck_this_tick = True
            else:
                agent.moved_this_tick = True
            if new_pos:
                if agent.carried_pod_id is not None:
                    pod = self.world.pod_state.get_pod(agent.carried_pod_id)
                    if pod:
                        pod.current_position = new_pos
                self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} moved to {new_pos}"
                    )

    def _detect_conflicts(self, tick: int, prev_positions: dict) -> int:
        """Detect vertex conflicts and oncoming (head-on swap) conflicts.

        参数
        ----------
        tick : int
            当前仿真 tick。
        prev_positions : dict[int, tuple[int, int]]
            Mapping of agent_id -> position *before* this tick's movement.

        返回
        ----
        int
            Number of positions where vertex conflicts occurred.
        """
        # --- 顶点冲突s: two agents on the same cell ---
        vertex_conflict_count = 0
        self.last_vertex_conflicts = []
        self.last_swap_conflicts = []
        self.world.traffic_vertex_conflicts_this_tick = 0
        self.world.traffic_swap_conflicts_this_tick = 0
        pos_to_agents: dict[tuple, list] = {}
        for agent in self.world.agents:
            pos_to_agents.setdefault(agent.position, []).append(agent.agent_id)

        for pos, agent_ids in pos_to_agents.items():
            if len(agent_ids) > 1:
                vertex_conflict_count += 1
                self.last_vertex_conflicts.append((pos, list(agent_ids)))
                ids_str = ", ".join(f"#{aid}" for aid in agent_ids)
                self.logger.warning(
                    f"[Tick {tick}] CONFLICT: Agents {ids_str} "
                    f"occupy the same cell {pos}"
                )

        # --- Oncoming (head-on / swap) conflicts ---
        # Two agents swap positions: A was at X and moved to Y while
        # B was at Y and moved to X.  This means they crossed the same
        # edge in opposite directions during this tick.
        agents = self.world.agents
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                a_prev = prev_positions[a.agent_id]
                b_prev = prev_positions[b.agent_id]
                # Check if they swapped (and actually moved)
                if (
                    a.position == b_prev
                    and b.position == a_prev
                    and a_prev != a.position  # A actually moved
                ):
                    self.last_swap_conflicts.append(
                        (a_prev, b_prev, a.agent_id, b.agent_id)
                    )
                    self.logger.warning(
                        f"[Tick {tick}] ONCOMING CONFLICT: "
                        f"Agent #{a.agent_id} ({a_prev}->{a.position}) and "
                        f"Agent #{b.agent_id} ({b_prev}->{b.position}) "
                        f"swapped positions (head-on collision)"
                    )

        self.world.traffic_vertex_conflicts_this_tick = int(
            vertex_conflict_count
        )
        self.world.traffic_swap_conflicts_this_tick = int(
            len(self.last_swap_conflicts)
        )
        self.total_vertex_conflicts_detected += int(vertex_conflict_count)
        self.total_swap_conflicts_detected += int(
            len(self.last_swap_conflicts)
        )
        return vertex_conflict_count

    def _handle_actions(self, tick: int):
        """Handle pickup, delivery, and return actions with configurable delays."""
        sim = self.config.simulation

        for agent in self.world.agents:
            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None:
                continue

            if agent.status == AgentStatus.EXITING:
                continue

            # WAITING_ASSIGNED is a runtime admission state, not an action
            # countdown.  Its DELIVER task intentionally remains ASSIGNED
            # until a station token and a path are acquired in
            # ``_plan_and_activate``.  Keep this branch explicit so a future
            # change to ``AgentState.is_waiting`` or ``wait_ticks`` cannot
            # accidentally execute the delivery action while the robot is
            # still waiting for admission.
            if agent.status == AgentStatus.WAITING_ASSIGNED:
                continue

            # --- Countdown in progress: decrement and skip ---
            if agent.is_waiting:
                agent.wait_ticks -= 1
                if agent.wait_ticks > 0:
                    continue
                # Countdown just finished — fall through to perform the action
            else:
                # --- Not waiting: check if agent just arrived ---
                if agent.position != active_task.destination:
                    continue
                if agent.has_path:
                    continue  # Still moving

                agent.stuck_this_tick = False

                # --- Pod availability check for PICK tasks ---
                if active_task.task_type == TaskType.PICK:
                    pod = self.world.pod_state.get_pod(active_task.pod_id)
                    if (pod is None
                            or pod.is_carried
                            or pod.current_position != agent.position):
                        self._handle_pod_unavailable(agent, active_task, tick)
                        continue

                # --- Determine required wait duration ---
                if active_task.task_type == TaskType.PICK:
                    required_wait = sim.pickup_duration
                elif active_task.task_type == TaskType.DELIVER:
                    required_wait = sim.station_process_duration
                elif active_task.task_type == TaskType.RETURN:
                    required_wait = sim.dropoff_duration
                else:
                    required_wait = 0

                # --- Start countdown ---
                if required_wait > 0:
                    if active_task.task_type == TaskType.DELIVER:
                        agent.status = AgentStatus.DELIVERING
                    agent.wait_ticks = required_wait
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} waiting "
                        f"{required_wait} ticks for {active_task.task_type.name} "
                        f"at {agent.position}"
                    )
                    continue  # Come back next tick

            # --- Perform the action ---
            if active_task.task_type == TaskType.PICK:
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    pod.pick_up(agent.agent_id)
                    agent.carried_pod_id = pod.pod_id
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} picked up "
                        f"Pod #{pod.pod_id} at {agent.position}"
                    )
                active_task.status = TaskStatus.COMPLETED
                active_task.completed_at = tick
                agent.clear_path()
                next_task = self.world.task_state.get_next_task_for_agent(
                    agent.agent_id
                )
                if (
                    self._uses_explicit_station_waiting()
                    and next_task is not None
                    and next_task.task_type == TaskType.DELIVER
                ):
                    queue = self.world.station_state.get_queue(
                        next_task.station_id
                    )
                    if queue is not None and queue.entry_position is not None:
                        # DELIVER remains ASSIGNED until a station token is
                        # actually promoted on the next planning phase.
                        self._mark_station_waiting(
                            agent, queue, next_task, tick
                        )

            elif active_task.task_type == TaskType.DELIVER:
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} delivered "
                        f"Pod #{pod.pod_id} to station at {agent.position}"
                    )
                    order = self.world.order_state.orders.get(active_task.order_id)
                    if order:
                        for sku, demand in order.sku_demands.items():
                            if sku in pod.sku_inventory:
                                pod.sku_inventory[sku] = max(
                                    0, pod.sku_inventory[sku] - demand
                                )
                        order.mark_pod_delivered(active_task.pod_id)

                agent.status = AgentStatus.EXITING

            elif active_task.task_type == TaskType.RETURN:
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    drop_pos = active_task.destination
                    pod.put_down(drop_pos)
                    agent.carried_pod_id = None
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} returned "
                        f"Pod #{pod.pod_id} to {drop_pos}"
                    )
                active_task.status = TaskStatus.COMPLETED
                active_task.completed_at = tick
                agent.clear_path()
                agent.assigned_task_id = None
                agent.clear_station_waiting()

                # Check if agent has more tasks
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is None:
                    agent.status = AgentStatus.IDLE

    def _check_order_completion(self, tick: int):
        """Check and update order completion status."""
        for order in self.world.order_state.get_in_progress_orders():
            if not self.world.task_state.all_order_tasks_completed(order.order_id):
                continue

            if order.is_fully_delivered:
                order.status = OrderStatus.COMPLETED
                order.completed_at = tick
                self.logger.info(
                    f"[Tick {tick}] Order #{order.order_id} COMPLETED "
                    f"(created at tick {order.created_at}, "
                    f"duration={tick - order.created_at} ticks)"
                )
                # Free up all agents that worked on this order
                order_tasks = self.world.task_state.get_tasks_for_order(order.order_id)
                agent_ids = {t.agent_id for t in order_tasks if t.agent_id is not None}
                for aid in agent_ids:
                    agent = self.world.get_agent(aid)
                    if agent.assigned_task_id is None and not agent.is_idle:
                        next_task = self.world.task_state.get_next_task_for_agent(aid)
                        if next_task is None:
                            agent.status = AgentStatus.IDLE
            else:
                order.status = OrderStatus.PENDING
                order.pod_ids.clear()
                self.logger.info(
                    f"[Tick {tick}] Order #{order.order_id} reset to PENDING "
                    f"(some pods were cancelled, retrying)"
                )

    def _print_summary(self):
        """Print simulation summary on shutdown."""
        self.logger.info("=" * 60)
        self.logger.info("SIMULATION SUMMARY")
        self.logger.info(f"  Total ticks:      {self.world.tick}")
        self.logger.info(f"  Total orders:     {self.world.order_state.total_orders}")
        self.logger.info(f"  Completed orders: {self.world.order_state.total_completed}")
        in_progress = len(self.world.order_state.get_in_progress_orders())
        pending = len(self.world.order_state.get_pending_orders())
        self.logger.info(f"  In-progress:      {in_progress}")
        self.logger.info(f"  Pending:          {pending}")

        summary = self.metrics.summarize()
        if summary:
            self.logger.info("-" * 40)
            self.logger.info("METRICS")
            self.logger.info(f"  Throughput (final):          {summary['final_throughput']}")
            self.logger.info(f"  Throughput (avg/100 ticks):  {summary['avg_throughput_per_100tick']:.2f}")
            self.logger.info(f"  Deadlock events (ticks):     {summary['total_deadlock_events']}")
            self.logger.info(f"  Congestion events (total):   {summary['total_congestion_events']}")
            self.logger.info(f"  Avg planning time:           {summary['avg_plan_ms']:.2f} ms")
            self.logger.info(f"  Avg assignment time:         {summary['avg_assign_ms']:.2f} ms")

        self.logger.info("=" * 60)

    def _record_path_len(self, agent, path):
        """Record planned path length on the agent's active task."""
        task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
        if task is not None and task.planned_path_len_at_start is None:
            task.planned_path_len_at_start = len(path)

    def _estimate_free_flow(self, task) -> int:
        """Manhattan distance + service time as free-flow estimate."""
        sr, sc = task.source
        dr, dc = task.destination
        manhattan = abs(sr - dr) + abs(sc - dc)
        sim = self.config.simulation
        if task.task_type == TaskType.PICK:
            service = sim.pickup_duration
        elif task.task_type == TaskType.DELIVER:
            service = sim.station_process_duration
        elif task.task_type == TaskType.RETURN:
            service = sim.dropoff_duration
        else:
            service = 0
        return manhattan + service

    # ------------------------------------------------------------------
    # Station queue helpers
    # ------------------------------------------------------------------

    def _check_queue_arrivals(self, tick: int, absorbed_entries: set):
        """Absorb CARRYING agents at entry_position into queue slots."""
        for agent in self.world.agents:
            if agent.status != AgentStatus.CARRYING:
                continue

            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None or active_task.task_type != TaskType.DELIVER:
                continue

            queue = self.world.station_state.get_queue(active_task.station_id)
            if queue is None or queue.entry_position is None:
                continue

            if agent.position != queue.entry_position:
                continue

            if agent.has_path:
                agent.clear_path()

            if queue.entry_position in absorbed_entries:
                continue

            if queue.check_in_from_entry(agent.agent_id, self.world):
                absorbed_entries.add(queue.entry_position)
                self.logger.info(
                    f"[Tick {tick}] Agent #{agent.agent_id} checked into queue "
                    f"at station {active_task.station_id} (entry={agent.position})"
                )

    def _clear_stale_entry_paths(self, tick: int):
        """Clear stale paths for DELIVER agents near a free entry.

        The space-time path planner may schedule waits or detours because
        it predicted the entry would be occupied.  When the entry has
        since become free, the stale path wastes ticks.  Clearing it
        lets _plan_and_activate re-plan a direct route this same tick.
        For adjacent agents, assigns a direct 1-step path to entry.
        """
        station_state = self.world.station_state
        claimed_entries = set()

        for agent in self.world.agents:
            if agent.status != AgentStatus.CARRYING or not agent.has_path:
                continue
            active_task = self.world.task_state.get_active_task_for_agent(
                agent.agent_id
            )
            if active_task is None or active_task.task_type != TaskType.DELIVER:
                continue
            queue = station_state.get_queue(active_task.station_id)
            if queue is None or queue.entry_position is None:
                continue

            remaining = agent.path[agent.path_index:]
            if not remaining or remaining[-1] != queue.entry_position:
                continue

            er, ec = queue.entry_position
            ar, ac = agent.position
            manhattan = abs(er - ar) + abs(ec - ac)
            if manhattan > 3:
                continue

            if len(remaining) <= manhattan:
                continue

            entry_pos = queue.entry_position
            if entry_pos in claimed_entries:
                continue

            entry_blocked = False
            for other in self.world.agents:
                if other.agent_id == agent.agent_id:
                    continue
                if other.position == entry_pos:
                    entry_blocked = True
                    break
                if other.has_path and other.path_index < len(other.path):
                    if other.path[other.path_index] == entry_pos:
                        entry_blocked = True
                        break
            if entry_blocked:
                continue

            agent.clear_path()
            self.logger.debug(
                f"[Tick {tick}] Cleared stale path for Agent #{agent.agent_id} "
                f"near entry {entry_pos} (dist={manhattan}, "
                f"remaining={len(remaining)} steps)"
            )

    def _process_station_exits(self, tick: int):
        """Two-phase exit: service->exit (1 step), then finalize next tick."""
        for agent in self.world.agents:
            if agent.status != AgentStatus.EXITING:
                continue
            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None:
                continue
            queue = self.world.station_state.get_queue(active_task.station_id)
            if queue is None:
                continue

            if queue.exit_position and agent.position == queue.exit_position:
                active_task.status = TaskStatus.COMPLETED
                active_task.completed_at = tick
                agent.clear_path()
                agent.status = AgentStatus.CARRYING
                agent.assigned_task_id = None
                agent.clear_station_waiting()
                self.logger.info(
                    f"[Tick {tick}] Agent #{agent.agent_id} finalized exit "
                    f"at station {active_task.station_id}, pos={agent.position}"
                )
                continue

            if queue.release_to_exit(agent.agent_id, self.world):
                self.logger.info(
                    f"[Tick {tick}] Agent #{agent.agent_id} moved to exit "
                    f"at station {active_task.station_id}, pos={agent.position}"
                )

    def _nudge_idle_neighbors(self, stuck_agent, tick: int):
        """Nudge idle agents adjacent to stuck_agent so it can leave a handoff cell."""
        ms = self.world.map_state
        occupied = {a.position for a in self.world.agents}
        pod_positions = {
            p.current_position
            for p in self.world.pod_state.pods.values()
            if not p.is_carried
        }
        sr, sc = stuck_agent.position
        for other in self.world.agents:
            if other.agent_id == stuck_agent.agent_id:
                continue
            if other.status != AgentStatus.IDLE or other.has_path:
                continue
            odr = abs(other.position[0] - sr)
            odc = abs(other.position[1] - sc)
            if odr + odc > 2:
                continue
            for nr, nc in ms.get_neighbors(other.position[0], other.position[1]):
                if not ms.is_walkable(nr, nc):
                    continue
                if (nr, nc) in occupied or (nr, nc) in pod_positions:
                    continue
                other.assign_path([(nr, nc)])
                occupied.discard(other.position)
                occupied.add((nr, nc))
                self.logger.info(
                    f"[Tick {tick}] Nudge: Agent #{other.agent_id} "
                    f"{other.position} -> ({nr},{nc}) to clear handoff "
                    f"for Agent #{stuck_agent.agent_id}"
                )
                break

    # ------------------------------------------------------------------
    # Pod unavailability handling
    # ------------------------------------------------------------------

    def _handle_pod_unavailable(self, agent, pick_task, tick: int):
        """Handle the case where a robot arrives but the pod is gone or taken."""
        task_state = self.world.task_state

        pick_task.status = TaskStatus.CANCELLED
        related = task_state.get_related_chain_tasks(pick_task)
        original_dispatch_sequence = next(
            (
                int(task.station_dispatch_sequence)
                for task in related
                if task.task_type == TaskType.DELIVER
                and getattr(task, "station_dispatch_sequence", None) is not None
            ),
            None,
        )
        for t in related:
            t.status = TaskStatus.CANCELLED

        self.logger.warning(
            f"[Tick {tick}] Agent #{agent.agent_id}: Pod #{pick_task.pod_id} "
            f"unavailable at {agent.position}. "
            f"Cancelled {1 + len(related)} tasks."
        )

        order = self.world.order_state.orders.get(pick_task.order_id)
        if order is None:
            self._reset_agent_to_idle(agent)
            return

        reserved_pods = {
            t.pod_id
            for t in task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        alt_pod = self._find_alternative_pod(
            order, pick_task.pod_id, reserved_pods
        )
        if alt_pod is None:
            if pick_task.pod_id in order.pod_ids:
                order.pod_ids.remove(pick_task.pod_id)
            self.logger.info(
                f"[Tick {tick}] Agent #{agent.agent_id}: No alternative pod "
                f"for Order #{order.order_id}. Setting agent to IDLE."
            )
            self._reset_agent_to_idle(agent)
            return

        station_pos = self.world.station_state.get_service_position(
            order.station_id
        )
        if station_pos is None:
            station_pos = self.world.map_state.station_positions.get(
                order.station_id
            )
        if station_pos is None:
            self._reset_agent_to_idle(agent)
            return

        exit_pos = self.world.station_state.get_exit_position(order.station_id)
        return_source = exit_pos or station_pos

        pod_return_planner = self.task_assigner.pod_return_planner
        if pod_return_planner is not None:
            return_dest = pod_return_planner.plan_return(
                alt_pod, return_source, self.world
            )
        else:
            return_dest = alt_pod.home_position

        new_pick = Task(
            task_type=TaskType.PICK,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=agent.position,
            destination=alt_pod.current_position,
        )
        new_pick.agent_id = agent.agent_id
        new_pick.status = TaskStatus.ASSIGNED
        new_pick.created_at = tick
        new_pick.assigned_at = tick
        new_pick.free_flow_time = self._estimate_free_flow(new_pick)

        new_deliver = Task(
            task_type=TaskType.DELIVER,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=alt_pod.current_position,
            destination=station_pos,
        )
        new_deliver.agent_id = agent.agent_id
        new_deliver.status = TaskStatus.ASSIGNED
        new_deliver.station_id = order.station_id
        new_deliver.station_dispatch_sequence = original_dispatch_sequence
        new_deliver.created_at = tick
        new_deliver.assigned_at = tick
        new_deliver.free_flow_time = self._estimate_free_flow(new_deliver)

        new_return = Task(
            task_type=TaskType.RETURN,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=return_source,
            destination=return_dest,
        )
        new_return.agent_id = agent.agent_id
        new_return.status = TaskStatus.ASSIGNED
        new_return.station_id = order.station_id
        new_return.created_at = tick
        new_return.assigned_at = tick
        new_return.free_flow_time = self._estimate_free_flow(new_return)

        if self._uses_dispatch_priority_station_admission():
            self._stamp_station_dispatch_sequences(
                (new_pick, new_deliver, new_return)
            )

        task_state.add_task(new_pick)
        task_state.add_task(new_deliver)
        task_state.add_task(new_return)

        if pick_task.pod_id in order.pod_ids:
            idx = order.pod_ids.index(pick_task.pod_id)
            order.pod_ids[idx] = alt_pod.pod_id

        agent.clear_path()
        agent.assigned_task_id = None
        agent.status = AgentStatus.IDLE
        agent.wait_ticks = 0

        self.logger.info(
            f"[Tick {tick}] Agent #{agent.agent_id}: Assigned alternative "
            f"Pod #{alt_pod.pod_id} for Order #{order.order_id}"
        )

    def _find_alternative_pod(self, order, original_pod_id, reserved_pods):
        """Find an alternative pod that satisfies at least some SKU demands."""
        needed_skus = {
            sku for sku, qty in order.sku_demands.items() if qty > 0
        }
        available_pods = self.world.pod_state.get_available_pods()
        fulfilled_pods = fulfilled_pod_ids_for_order(self.world, order)

        best_pod = None
        best_score = 0
        for pod in available_pods:
            if pod.pod_id in reserved_pods or pod.is_carried:
                continue
            if int(pod.pod_id) in fulfilled_pods:
                continue
            if pod.pod_id == original_pod_id:
                continue
            score = sum(
                1 for sku in needed_skus
                if sku in pod.sku_inventory and pod.sku_inventory[sku] > 0
            )
            if score > best_score:
                best_score = score
                best_pod = pod
        return best_pod

    def _reset_agent_to_idle(self, agent):
        """Reset an agent to IDLE state."""
        self.world.station_state.cancel_waiting_reservation(agent.agent_id)
        agent.clear_path()
        agent.assigned_task_id = None
        agent.status = AgentStatus.IDLE
        agent.wait_ticks = 0
        agent.carried_pod_id = None
        agent.clear_station_waiting()
