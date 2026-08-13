"""Wave-based admission control: a per-replica ledger that gates new
conversation starts by estimated free KV budget, with reactive migration.

``AdmissionLedger`` runs as one Ray actor so every ``AgentLoopWorker`` shares
the same ledger state.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time

import ray

from llm_d_rl_verl_integration.p2p_addressing import p2p_listener_host

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class GrowthEstimator:
    """Causal estimate of a conversation's remaining KV growth.

    reserve_mode keys on turn index, context-size bucket, or both; reserve_z
    adds a std margin. Only fed from already-completed conversations.
    """

    def __init__(
        self,
        initial_guess: float,
        prior_weight: float,
        reserve_mode: str = "size",
        reserve_z: float = 0.0,
    ):
        if reserve_mode not in ("turn", "size", "turn_size"):
            raise ValueError(f"unknown reserve_mode: {reserve_mode!r}")
        self._initial_guess = initial_guess
        self._prior_weight = prior_weight
        self._reserve_mode = reserve_mode
        self._reserve_z = reserve_z

        # Prior-blended running mean (unchanged design from the turn-only
        # estimator), keyed per reserve_mode.
        self._growth_mean: dict = {}
        self._growth_mean_n: dict = {}
        # Separate, unblended Welford running mean/variance over REAL samples
        # only, used purely for the std margin.
        self._growth_rmean: dict = {}
        self._growth_m2: dict = {}
        self._growth_count: dict = {}

    @staticmethod
    def _size_bucket(size: float) -> int:
        """log2 bucket of current context size - coarse enough that a few
        hundred conversations still land enough real samples per bucket."""
        return int(math.log2(max(size, 1)))

    def key(self, context_size: float, turn_index: int):
        """Estimator key for this resident, per reserve_mode."""
        if self._reserve_mode == "turn":
            return turn_index
        if self._reserve_mode == "size":
            return self._size_bucket(context_size)
        return (turn_index, self._size_bucket(context_size))  # "turn_size"

    def _mean(self, key) -> float:
        return self._growth_mean.get(key, self._initial_guess)

    def _std(self, key) -> float:
        n = self._growth_count.get(key, 0)
        if n < 2:
            return 0.0
        return (self._growth_m2[key] / (n - 1)) ** 0.5

    def estimate(self, key) -> float:
        return self._mean(key) + self._reserve_z * self._std(key)

    def observe(self, key, remaining_growth: float) -> None:
        n = self._growth_mean_n.get(key, self._prior_weight)
        old_mean = self._mean(key)
        self._growth_mean[key] = old_mean + (remaining_growth - old_mean) / (n + 1)
        self._growth_mean_n[key] = n + 1

        c = self._growth_count.get(key, 0) + 1
        rold = self._growth_rmean.get(key, 0.0)
        rnew = rold + (remaining_growth - rold) / c
        self._growth_m2[key] = self._growth_m2.get(key, 0.0) + (remaining_growth - rold) * (remaining_growth - rnew)
        self._growth_rmean[key] = rnew
        self._growth_count[key] = c


@ray.remote
class AdmissionLedger:
    """Admission state shared by every ``WaveAdmissionLLMClient``.

    Prefers staying resident once admitted; with ``allow_reactive_migration``
    it falls back to another replica only when the resident one cannot fit the
    next turn.
    """

    def __init__(
        self,
        replicas: list[str],
        *,
        budget_tokens_per_replica: float,
        wave1_size: int,
        initial_growth_guess: float,
        prior_weight: float,
        max_wait_s: float,
        poll_interval_s: float,
        allow_reactive_migration: bool = True,
        reserve_mode: str = "size",
        reserve_z: float = 0.0,
        migration_cost_ratio: float = 1.0,
        p2p_kv_available: bool = False,
        p2p_port: int = 7777,
        migration_cost_ratio_p2p: float = 0.0,
        p2p_nosidecar: bool = False,
        oracle_reserve: bool = False,
        lpt_window_s: float = 0.0,
        rebalance_slack: float = -1.0,
    ):
        self._replicas = list(replicas)
        self._budget = budget_tokens_per_replica
        self._wave1_size = wave1_size
        self._max_wait_s = max_wait_s
        self._poll_interval_s = poll_interval_s
        self._allow_reactive_migration = allow_reactive_migration
        self._migration_cost_ratio = migration_cost_ratio
        # With P2P available a migration can pull the resident's KV instead of
        # recomputing it, so migration_cost_ratio_p2p (default 0.0) applies.
        self._p2p_kv_available = p2p_kv_available
        self._p2p_port = p2p_port
        self._migration_cost_ratio_p2p = migration_cost_ratio_p2p
        self._p2p_nosidecar = p2p_nosidecar

        self._used: dict[str, float] = {r: 0.0 for r in replicas}
        self._estimator = GrowthEstimator(
            initial_growth_guess, prior_weight, reserve_mode=reserve_mode, reserve_z=reserve_z,
        )

        # request_id -> assigned replica (residency, sticky for the trajectory).
        self._resident: dict[str, str] = {}
        # request_id -> {turn_index: context_size measured after that turn}.
        self._history: dict[str, dict[int, float]] = {}
        # request_id -> current estimated remaining growth (cached from the
        # estimator at the request's most-recently-observed turn index).
        self._reserve_charge: dict[str, float] = {}

        self._admitted_count = 0
        self._delayed_polls = 0
        self._forced_admissions = 0
        self._migrations = 0

        # Oracle reserve: charge the trace's true remaining tokens instead of the
        # estimator's guess. Only meaningful for a replay (the size is known up
        # front); it measures what perfect prediction would be worth.
        self._oracle_reserve = oracle_reserve
        self._oracle_final: dict[str, float] = {}

        # Windowed LPT. Every session of a rollout batch calls acquire() within a
        # few ms, so a short collection window sees essentially the whole batch;
        # sorting it longest-work-first before placing is the classic ~4/3
        # makespan heuristic, versus placing in arrival order (2-approx).
        self._lpt_window_s = lpt_window_s
        self._lpt_pending: list[tuple[float, str, float]] = []   # (-work, rid, context)
        self._lpt_result: dict[str, str] = {}
        self._lpt_event: "asyncio.Event | None" = None
        self._lpt_closed = False
        self._lpt_batches = 0

        # Imbalance-driven migration. The shipped rule is FULLNESS-driven: a
        # session moves only when its resident replica cannot fit the next turn,
        # so it stays put while another replica sits far emptier. With
        # rebalance_slack >= 0 a session also moves when some other replica has
        # more than slack*budget additional free budget, targeting the straggler
        # directly instead of waiting for a replica to fill up. Negative disables.
        self._rebalance_slack = rebalance_slack
        self._rebalances = 0

        logger.info(
            "[AdmissionLedger] %d replicas, budget=%.0f tok/replica, wave1_size=%d, "
            "allow_reactive_migration=%s, reserve_mode=%s, reserve_z=%.1f, migration_cost_ratio=%.1f, "
            "p2p_kv_available=%s, migration_cost_ratio_p2p=%.2f, p2p_nosidecar=%s, p2p_port=%d",
            len(replicas), budget_tokens_per_replica, wave1_size, allow_reactive_migration,
            reserve_mode, reserve_z, migration_cost_ratio,
            p2p_kv_available, migration_cost_ratio_p2p, p2p_nosidecar, p2p_port,
        )

    def _charge(self, request_id: str, context_size: float, turn_index: int) -> float:
        """Reserve charge for a resident: predicted remaining growth.

        Oracle mode substitutes the trace's true final context for the estimator,
        which turns this into a measurement of what perfect prediction is worth.
        """
        if self._oracle_reserve:
            final = self._oracle_final.get(request_id)
            if final is not None:
                return max(0.0, final - context_size)
        return self._estimator.estimate(self._estimator.key(context_size, turn_index))

    def _reserve(self, replica: str) -> float:
        return sum(
            self._reserve_charge.get(rid, 0.0)
            for rid, r in self._resident.items()
            if r == replica
        )

    def _estimated_free(self, replica: str) -> float:
        return self._budget - self._used[replica] - self._reserve(replica)

    def _least_loaded(self) -> str:
        return max(self._replicas, key=self._estimated_free)

    def _book(self, request_id: str, replica: str, turn_index: int, context_size: float) -> None:
        """Record a booking for this request on `replica` at `turn_index`."""
        history = self._history.setdefault(request_id, {})
        if self._resident.get(request_id) == replica and (turn_index - 1) in history:
            prev = history[turn_index - 1]
            self._used[replica] += max(0.0, context_size - prev)
        else:
            self._used[replica] += context_size
        self._resident[request_id] = replica
        history[turn_index] = context_size
        self._reserve_charge[request_id] = self._charge(request_id, context_size, turn_index)

    def _release_booking(self, request_id: str, replica: str, turn_index: int) -> None:
        """Release the booking recorded for request_id at turn_index on
        replica (used when migrating a resident conversation away)."""
        size = self._history.get(request_id, {}).get(turn_index, 0.0)
        self._used[replica] = max(0.0, self._used[replica] - size)

    async def acquire(
        self,
        request_id: str,
        *,
        turn_index: int,
        context_size: float,
        session_final: float | None = None,
        session_work: float | None = None,
    ) -> dict:
        """Pick a replica: {"replica": addr, "kv_source": addr-or-None}.

        session_final/session_work are trace-derived hints (final context size and
        total prefill+decode work). Only the replay client sends them; every other
        caller leaves them None and the estimator is used as before.
        """
        if session_final is not None:
            self._oracle_final[request_id] = float(session_final)
        if turn_index > 0:
            return self._continue_or_migrate(request_id, turn_index, context_size)

        if self._lpt_window_s > 0:
            replica = await self._lpt_place(request_id, context_size, session_work)
            self._book(request_id, replica, 0, context_size)
            return {"replica": replica, "kv_source": None}

        self._admitted_count += 1
        if self._admitted_count <= self._wave1_size:
            replica = self._least_loaded()
        else:
            deadline = time.monotonic() + self._max_wait_s
            replica = None
            while time.monotonic() < deadline:
                candidate = self._least_loaded()
                if self._estimated_free(candidate) >= context_size:
                    replica = candidate
                    break
                self._delayed_polls += 1
                await asyncio.sleep(self._poll_interval_s)
            if replica is None:
                # Safety valve: never hang a training run forever waiting for
                # headroom that may never appear (mirrors the simulator's own
                # deadlock-avoidance concern, FINDINGS.md #1).
                self._forced_admissions += 1
                replica = self._least_loaded()
                logger.warning(
                    "[AdmissionLedger] forced admission of %s after %.1fs wait "
                    "(no replica reached estimated_free>=%.0f)",
                    request_id, self._max_wait_s, context_size,
                )

        self._book(request_id, replica, 0, context_size)
        return {"replica": replica, "kv_source": None}

    async def _lpt_place(self, request_id: str, context_size: float, session_work) -> str:
        """Windowed longest-processing-time-first placement.

        Collect turn-0 arrivals for lpt_window_s, then assign them largest-work
        first, each to the replica with the most estimated free budget. Sessions
        arriving after the window fall through to plain least-loaded so a late or
        retried trajectory can never stall.
        """
        if self._lpt_closed:
            return self._least_loaded()
        work = float(session_work) if session_work else context_size
        self._lpt_pending.append((-work, request_id, context_size))
        first = self._lpt_event is None
        if first:
            self._lpt_event = asyncio.Event()
        ev = self._lpt_event
        if first:
            await asyncio.sleep(self._lpt_window_s)
            batch = sorted(self._lpt_pending)
            self._lpt_pending = []
            # Greedy LPT: largest first onto the emptiest replica, charging each
            # assignment as we go so later items see the updated load.
            provisional: dict[str, float] = {r: 0.0 for r in self._replicas}
            for negw, rid, ctx in batch:
                target = max(self._replicas, key=lambda r: self._estimated_free(r) - provisional[r])
                provisional[target] += -negw
                self._lpt_result[rid] = target
            self._lpt_batches += 1
            self._lpt_closed = True
            logger.info(
                "[AdmissionLedger] LPT batch %d: placed %d sessions in a %.2fs window "
                "(work range %.0f..%.0f tokens)",
                self._lpt_batches, len(batch), self._lpt_window_s,
                -batch[-1][0] if batch else 0, -batch[0][0] if batch else 0,
            )
            ev.set()
        else:
            try:
                await asyncio.wait_for(ev.wait(), timeout=self._lpt_window_s + 30.0)
            except asyncio.TimeoutError:
                logger.warning("[AdmissionLedger] LPT window timed out for %s - least_loaded", request_id)
                return self._least_loaded()
        return self._lpt_result.pop(request_id, None) or self._least_loaded()

    def _continue_or_migrate(self, request_id: str, turn_index: int, context_size: float) -> dict:
        resident = self._resident.get(request_id)
        if resident is None:
            # Shouldn't happen (turn 0 always assigns first) but fail safe
            # rather than crash a training run.
            replica = self._least_loaded()
            self._book(request_id, replica, turn_index, context_size)
            return {"replica": replica, "kv_source": None}

        prev_size = self._history.get(request_id, {}).get(turn_index - 1, 0.0)
        incremental_need = max(0.0, context_size - prev_size)

        if self._rebalance_slack >= 0.0 and self._allow_reactive_migration:
            others = [g for g in self._replicas if g != resident]
            cand = max(others, key=self._estimated_free, default=None)
            if cand is not None:
                gain = self._estimated_free(cand) - self._estimated_free(resident)
                if gain > self._rebalance_slack * self._budget and self._estimated_free(cand) >= context_size:
                    self._rebalances += 1
                    self._release_booking(request_id, resident, turn_index - 1)
                    self._book(request_id, cand, turn_index, context_size)
                    kv_source = None
                    if self._p2p_kv_available:
                        kv_source = f"{p2p_listener_host(self._replicas.index(resident))}:{self._p2p_port}"
                    return {"replica": cand, "kv_source": kv_source}

        if self._estimated_free(resident) >= incremental_need:
            self._book(request_id, resident, turn_index, context_size)
            return {"replica": resident, "kv_source": None}

        # Migrate only when the deficit is large relative to the cost it
        # relieves: a full re-prefill without P2P, a cheap pull with it.
        effective_cost_ratio = (
            self._migration_cost_ratio_p2p if self._p2p_kv_available else self._migration_cost_ratio
        )
        deficit = incremental_need - self._estimated_free(resident)
        migration_worth_it = (
            self._allow_reactive_migration
            and deficit >= effective_cost_ratio * context_size
        )
        if migration_worth_it:
            others = [g for g in self._replicas if g != resident]
            target = max(others, key=self._estimated_free, default=None)
            if target is not None and self._estimated_free(target) >= context_size:
                self._migrations += 1
                self._release_booking(request_id, resident, turn_index - 1)
                self._book(request_id, target, turn_index, context_size)
                kv_source = None
                if self._p2p_kv_available:
                    # The source's P2P control socket, not its HTTP address.
                    kv_source = (
                        f"{p2p_listener_host(self._replicas.index(resident))}"
                        f":{self._p2p_port}"
                    )
                logger.info(
                    "[AdmissionLedger] migrated %s: %s -> %s at turn %d "
                    "(deficit %.0f >= %.2fx migration cost %.0f, kv_source=%s)",
                    request_id, resident, target, turn_index,
                    deficit, effective_cost_ratio, context_size, kv_source,
                )
                return {"replica": target, "kv_source": kv_source}

        # Nothing fits or migration isn't worth it: stay resident and overshoot
        # rather than stall a running conversation.
        self._book(request_id, resident, turn_index, context_size)
        return {"replica": resident, "kv_source": None}

    def record_turn(self, request_id: str, *, turn_index: int, context_size: float) -> None:
        """Record the context size measured after a completed turn."""
        replica = self._resident.get(request_id)
        if replica is None:
            return
        history = self._history.setdefault(request_id, {})
        prev_size = history.get(turn_index, context_size)
        delta = context_size - prev_size
        if delta:
            self._used[replica] += delta
        history[turn_index] = context_size
        self._reserve_charge[request_id] = self._charge(request_id, context_size, turn_index)

    def on_trajectory_done(self, request_id: str) -> None:
        """Release bookings and feed the estimator with completed-turn growth."""
        replica = self._resident.pop(request_id, None)
        history = self._history.pop(request_id, {})
        self._reserve_charge.pop(request_id, None)
        if replica is None or not history:
            return
        final_size = history[max(history.keys())]
        for turn_idx, size_at_turn in history.items():
            key = self._estimator.key(size_at_turn, turn_idx)
            self._estimator.observe(key, final_size - size_at_turn)
        self._used[replica] = max(0.0, self._used[replica] - final_size)

    def stats(self) -> dict:
        return {
            "used": dict(self._used),
            "estimated_free": {r: self._estimated_free(r) for r in self._replicas},
            "admitted": self._admitted_count,
            "rebalances": self._rebalances,
            "delayed_polls": self._delayed_polls,
            "forced_admissions": self._forced_admissions,
            "migrations": self._migrations,
            "resident_count": len(self._resident),
        }


def compute_budget_tokens_per_replica(
    *,
    gpu_capacity_gb: float | None = None,
    gpu_memory_utilization: float | None = None,
    weights_gb: float | None = None,
    bytes_per_token: float | None = None,
) -> float:
    """Per-GPU KV token budget; args fall back to VERL_WAVE_ADMISSION_* env."""
    if gpu_capacity_gb is None:
        gpu_capacity_gb = _env_float("WAVE_ADMISSION_GPU_CAPACITY_GB", 139.8)
    if gpu_memory_utilization is None:
        gpu_memory_utilization = _env_float("WAVE_ADMISSION_GPU_UTIL", 0.6)
    if weights_gb is None:
        weights_gb = _env_float("WAVE_ADMISSION_WEIGHTS_GB", 15.2)
    if bytes_per_token is None:
        bytes_per_token = _env_float("WAVE_ADMISSION_KV_BYTES_PER_TOKEN", 57344.0)

    usable_gb = max(0.0, gpu_memory_utilization * gpu_capacity_gb - weights_gb)
    return (usable_gb * 1e9) / bytes_per_token
