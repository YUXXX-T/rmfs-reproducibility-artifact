from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from WorldModel.data.fuse_and_split import (
    LEGACY_LYAPUNOV_COLLECTION_SCHEMA,
    _detect_lyapunov_collection_schema,
)


def _snapshot_v2():
    return {
        "schema_version": "lyapunov_l0_snapshot_v2",
        "arrival_manifest": [],
        "traffic_diagnostics": {},
    }


def _sample_v2():
    return {
        "lyapunov_l0_collection_schema_version": "lyapunov_l0_collection_v2",
        "lyapunov_l0_start": _snapshot_v2(),
        "lyapunov_l0_post_action": _snapshot_v2(),
        "lyapunov_l0_end": _snapshot_v2(),
    }


def _snapshot_v3():
    return {
        "schema_version": "lyapunov_l1_snapshot_v3",
        "arrival_manifest": [],
        "traffic_diagnostics": {},
    }


def _sample_v3():
    return {
        "lyapunov_l0_collection_schema_version": "lyapunov_l1_collection_v3",
        "lyapunov_l0_start": _snapshot_v3(),
        "lyapunov_l0_post_action": _snapshot_v3(),
        "lyapunov_l0_end": _snapshot_v3(),
    }


class FuseLyapunovSchemaTests(unittest.TestCase):
    def test_legacy_payload_is_classified_as_v1(self):
        schema = _detect_lyapunov_collection_schema(
            [{"lyapunov_l0_start": {}}], "legacy.pt"
        )
        self.assertEqual(schema, LEGACY_LYAPUNOV_COLLECTION_SCHEMA)

    def test_valid_v2_schema_is_accepted(self):
        schema = _detect_lyapunov_collection_schema(
            [_sample_v2()], "v2.pt"
        )
        self.assertEqual(schema, "lyapunov_l0_collection_v2")

    def test_valid_v3_schema_is_accepted(self):
        schema = _detect_lyapunov_collection_schema(
            [_sample_v3()], "v3.pt"
        )
        self.assertEqual(schema, "lyapunov_l1_collection_v3")

    def test_v2_v3_mix_inside_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixed Lyapunov"):
            _detect_lyapunov_collection_schema(
                [_sample_v2(), _sample_v3()], "mixed_versions.pt"
            )

    def test_v1_v2_mix_inside_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixed Lyapunov"):
            _detect_lyapunov_collection_schema(
                [_sample_v2(), {"lyapunov_l0_start": {}}], "mixed.pt"
            )

    def test_v2_requires_raw_arrival_and_traffic_fields(self):
        sample = _sample_v2()
        sample["lyapunov_l0_end"] = {
            "schema_version": "lyapunov_l0_snapshot_v2"
        }
        with self.assertRaisesRegex(ValueError, "arrival_manifest"):
            _detect_lyapunov_collection_schema([sample], "broken.pt")


if __name__ == "__main__":
    unittest.main()
