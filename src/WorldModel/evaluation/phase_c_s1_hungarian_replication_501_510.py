"""Isolated seed-block override for the 501--510 replication.

Only the formal seed block, output directory, schema identifiers, and
bootstrap RNG seed change.  The manifest source, policies, checkpoint,
configs, horizons, and pass criteria remain identical to the frozen 491--500
comparison.
"""

from __future__ import annotations

from WorldModel.evaluation import phase_c_s1_hungarian_protocol as protocol


SEEDS = tuple(range(501, 511))
OUTPUT_ROOT = (
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "phasec_s1_hungarian_replication_501_510_v1"
)

_ORIGINAL_FORMAL_PROTOCOL = protocol.formal_protocol
_INSTALLED = False


def install():
    """Patch the shared immutable implementation for this isolated process."""
    global _INSTALLED
    if _INSTALLED:
        return protocol

    protocol.SCHEMA_VERSION = (
        "phase_c_s1_hungarian_seed_replication_protocol_v1"
    )
    protocol.FROZEN_BUNDLE_SCHEMA_VERSION = (
        "phase_c_s1_hungarian_seed_replication_bundle_v1"
    )
    protocol.PER_SEED_SCHEMA_VERSION = (
        "phase_c_s1_hungarian_seed_replication_result_v1"
    )
    protocol.AGGREGATE_SCHEMA_VERSION = (
        "phase_c_s1_hungarian_seed_replication_aggregate_v1"
    )
    protocol.OUTPUT_ROOT = OUTPUT_ROOT
    protocol.SEEDS = SEEDS
    protocol.BOOTSTRAP_SEED = 20260730

    def replication_protocol() -> dict:
        payload = _ORIGINAL_FORMAL_PROTOCOL()
        payload["claim"] = (
            "untouched 501--510 seed replication of the frozen Phase-C/S1/"
            "Hungarian/Greedy comparison; only the seed block changes"
        )
        payload["formal_test"].update(
            {
                "replication_role": "new_seed_block_only",
                "reference_seed_block": list(range(491, 501)),
                "seed_change_only": True,
            }
        )
        return payload

    protocol.formal_protocol = replication_protocol
    _INSTALLED = True
    return protocol

