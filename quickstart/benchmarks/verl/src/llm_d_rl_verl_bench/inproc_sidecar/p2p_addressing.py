# P2P peer addressing, shared by p2p_replica.py (binds the listener) and
# wave_admission/admission.py (names the source). Stdlib only, so the
# AdmissionLedger actor can import it without the vLLM rollout stack.
from __future__ import annotations

DEFAULT_P2P_CONNECTOR_PORT = 7777

P2P_LOOPBACK_NET = "127.0.7."


def p2p_listener_host(replica_index: int) -> str:
    """IP that replica `replica_index` binds its P2P control socket on.

    One IP per replica, so the tier port stays flat. Single-node only: loopback
    is not routable off-box.
    """
    if replica_index < 0 or replica_index > 253:
        raise ValueError(
            f"replica_index {replica_index} outside the {P2P_LOOPBACK_NET}0/24 "
            "alias block; use real per-replica IPs for a fleet this large"
        )
    return f"{P2P_LOOPBACK_NET}{replica_index + 1}"
