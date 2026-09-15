"""Canonical output contract for :class:`LongRiskHead`.

Keep every runtime consumer on this schema instead of assigning semantic
names to positional head outputs independently.  The head predicts six raw
channels; the quantile combo is a derived signal and is not a seventh learned
channel.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, TypeVar


LONG_RISK_SCHEMA_VERSION = "long_risk_6d_quantile_combo_v2"

LONG_RISK_OUTPUT_NAMES = (
    "peak_q90",
    "peak_q95",
    "cvar_q90",
    "terminal_q90",
    "delta_group_q90",
    "event_logit",
)
LONG_RISK_OUTPUT_DIM = len(LONG_RISK_OUTPUT_NAMES)
LONG_RISK_OUTPUT_INDEX = {
    name: index for index, name in enumerate(LONG_RISK_OUTPUT_NAMES)
}

LONG_RISK_QUANTILE_COMBO_WEIGHTS = {
    "peak_q95": 0.2,
    "cvar_q90": 0.3,
    "terminal_q90": 0.5,
}

LONG_RISK_DRIFT_SIGNAL_KEYS = {
    "combo": "long_risk_quantile_combo",
    "terminal": "long_risk_terminal_q90",
    "event_logit": "long_risk_event_logit",
}
LONG_RISK_DRIFT_SIGNALS = tuple(LONG_RISK_DRIFT_SIGNAL_KEYS)


ScalarT = TypeVar("ScalarT")


def long_risk_quantile_combo(
    peak_q95: ScalarT,
    cvar_q90: ScalarT,
    terminal_q90: ScalarT,
):
    """Return the canonical explicit tail-quantile combination.

    The inputs may be Python scalars or tensor scalars. Keeping this formula
    here prevents training diagnostics, offline evaluation, and online
    inference from silently drifting to different weights.
    """

    return (
        LONG_RISK_QUANTILE_COMBO_WEIGHTS["peak_q95"] * peak_q95
        + LONG_RISK_QUANTILE_COMBO_WEIGHTS["cvar_q90"] * cvar_q90
        + LONG_RISK_QUANTILE_COMBO_WEIGHTS["terminal_q90"] * terminal_q90
    )


def long_risk_drift_signal_key(signal: str) -> str:
    """Return the decoded record key for a configured drift signal."""

    try:
        return LONG_RISK_DRIFT_SIGNAL_KEYS[signal]
    except KeyError as exc:
        valid = ", ".join(LONG_RISK_DRIFT_SIGNALS)
        raise ValueError(
            f"energy_drift_signal must be one of: {valid}"
        ) from exc


def decode_long_risk_predictions(
    predictions: Optional[Sequence[ScalarT]],
    *,
    scalar: Callable[[ScalarT], float] = float,
) -> Dict[str, float]:
    """Decode the six positional head outputs and derive the quantile combo.

    ``None`` means that the optional head was not evaluated and therefore
    produces a zero-valued record.  A present vector must satisfy the exact
    six-channel contract; silently accepting a shorter or longer vector would
    reintroduce the positional schema bug this helper is intended to prevent.
    """

    if predictions is None:
        values = [0.0] * LONG_RISK_OUTPUT_DIM
    else:
        if len(predictions) != LONG_RISK_OUTPUT_DIM:
            raise ValueError(
                "LongRiskHead output contract mismatch: expected "
                f"{LONG_RISK_OUTPUT_DIM} channels "
                f"{LONG_RISK_OUTPUT_NAMES}, got {len(predictions)}"
            )
        values = [scalar(value) for value in predictions]

    peak_q90 = float(values[LONG_RISK_OUTPUT_INDEX["peak_q90"]])
    peak_q95 = float(values[LONG_RISK_OUTPUT_INDEX["peak_q95"]])
    cvar_q90 = float(values[LONG_RISK_OUTPUT_INDEX["cvar_q90"]])
    terminal_q90 = float(values[LONG_RISK_OUTPUT_INDEX["terminal_q90"]])
    delta_group_q90 = float(
        values[LONG_RISK_OUTPUT_INDEX["delta_group_q90"]]
    )
    event_logit = float(values[LONG_RISK_OUTPUT_INDEX["event_logit"]])

    quantile_combo = long_risk_quantile_combo(
        peak_q95,
        cvar_q90,
        terminal_q90,
    )

    return {
        "long_risk_peak_q90": peak_q90,
        "long_risk_peak_q95": peak_q95,
        "long_risk_cvar_q90": cvar_q90,
        "long_risk_terminal_q90": terminal_q90,
        "long_risk_delta_group_q90": delta_group_q90,
        "long_risk_event_logit": event_logit,
        "long_risk_quantile_combo": quantile_combo,
        # Backward-compatible field name used by existing traces/protocols.
        # Its value is now the intended explicit quantile combination.
        "long_risk_combo_q90": quantile_combo,
    }


def long_risk_runtime_contract() -> Dict[str, object]:
    """Serializable contract for experiment metadata and audit reports."""

    return {
        "schema_version": LONG_RISK_SCHEMA_VERSION,
        "output_order": list(LONG_RISK_OUTPUT_NAMES),
        "quantile_combo_weights": dict(LONG_RISK_QUANTILE_COMBO_WEIGHTS),
        "drift_signal_keys": dict(LONG_RISK_DRIFT_SIGNAL_KEYS),
    }
