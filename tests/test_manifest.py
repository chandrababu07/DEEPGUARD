"""
Unit tests for DeepGuard manifest validation, path checking, and leakage-free splitting.
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.manifest import ManifestManager, ManifestRecord, ValidationReport


class TestManifestManager(unittest.TestCase):
    """Test suite for ManifestRecord, ManifestManager, and ValidationReport."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_manifest.csv")

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_write_and_read_manifest(self):
        """Test round-trip serialization and deserialization of manifest records."""
        records = [
            ManifestRecord(
                sample_id="SAMPLE_001",
                video_path="videos/real_01.mp4",
                label=0,
                dataset_name="FaceForensics++",
                split="train",
                manipulation_type="original",
                original_video_id="ACTOR_01",
                preprocessing_status="PENDING",
            ),
            ManifestRecord(
                sample_id="SAMPLE_002",
                video_path="videos/fake_01.mp4",
                label=1,
                dataset_name="FaceForensics++",
                split="train",
                manipulation_type="Deepfakes",
                original_video_id="ACTOR_01",
                preprocessing_status="PENDING",
            ),
        ]
        ManifestManager.write_manifest(records, self.csv_path)
        loaded = ManifestManager.read_manifest(self.csv_path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].sample_id, "SAMPLE_001")
        self.assertEqual(loaded[0].label, 0)
        self.assertEqual(loaded[1].sample_id, "SAMPLE_002")
        self.assertEqual(loaded[1].label, 1)
        self.assertEqual(loaded[1].manipulation_type, "Deepfakes")

    def test_detect_duplicate_sample_id(self):
        """Test that duplicate sample IDs are flagged as invalid."""
        records = [
            ManifestRecord("ID_001", "v1.mp4", 0, "test_ds", "train"),
            ManifestRecord("ID_001", "v2.mp4", 1, "test_ds", "train"),
        ]
        report = ManifestManager.validate_manifest(records, check_video_paths=False)

        self.assertFalse(report.is_valid)
        self.assertIn("ID_001", report.duplicate_sample_ids)
        self.assertEqual(report.invalid_records_count, 1)

    def test_detect_invalid_label(self):
        """Test that labels other than 0 and 1 are flagged as invalid."""
        records = [
            ManifestRecord("ID_001", "v1.mp4", 2, "test_ds", "train"),  # Invalid: 2
            ManifestRecord("ID_002", "v2.mp4", "invalid", "test_ds", "train"),  # Invalid: string
        ]
        report = ManifestManager.validate_manifest(records, check_video_paths=False)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.invalid_records_count, 2)
        error_fields = [e.field_name for e in report.errors]
        self.assertIn("label", error_fields)

    def test_detect_invalid_split(self):
        """Test that unsupported split names are flagged."""
        records = [
            ManifestRecord("ID_001", "v1.mp4", 0, "test_ds", "training_set"),  # Invalid
            ManifestRecord("ID_002", "v2.mp4", 1, "test_ds", "val"),  # Valid alias -> normalized to validation
        ]
        report = ManifestManager.validate_manifest(records, check_video_paths=False)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.valid_records_count, 1)  # ID_002 is normalized to validation
        self.assertEqual(report.invalid_records_count, 1)  # ID_001 is invalid

    def test_split_class_counts_reporting(self):
        """Test that class distribution per split is correctly calculated."""
        records = [
            ManifestRecord("S1", "v1.mp4", 0, "ds", "train"),
            ManifestRecord("S2", "v2.mp4", 0, "ds", "train"),
            ManifestRecord("S3", "v3.mp4", 1, "ds", "train"),
            ManifestRecord("S4", "v4.mp4", 0, "ds", "validation"),
            ManifestRecord("S5", "v5.mp4", 1, "ds", "validation"),
            ManifestRecord("S6", "v6.mp4", 1, "ds", "test"),
        ]
        report = ManifestManager.validate_manifest(records, check_video_paths=False)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.split_class_counts["train"]["real"], 2)
        self.assertEqual(report.split_class_counts["train"]["fake"], 1)
        self.assertEqual(report.split_class_counts["train"]["total"], 3)

        self.assertEqual(report.split_class_counts["validation"]["real"], 1)
        self.assertEqual(report.split_class_counts["validation"]["fake"], 1)
        self.assertEqual(report.split_class_counts["validation"]["total"], 2)

        self.assertEqual(report.split_class_counts["test"]["real"], 0)
        self.assertEqual(report.split_class_counts["test"]["fake"], 1)
        self.assertEqual(report.split_class_counts["test"]["total"], 1)

    def test_detect_missing_video_files(self):
        """Test physical video file existence check."""
        non_existent_file = os.path.join(self.temp_dir, "does_not_exist.mp4")
        records = [
            ManifestRecord("S1", non_existent_file, 0, "ds", "train"),
        ]
        report = ManifestManager.validate_manifest(records, check_video_paths=True)

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.missing_video_paths), 1)

    def test_leakage_detection_when_scattered(self):
        """Test that placing related original and fake videos in different splits is detected as leakage."""
        records = [
            ManifestRecord("S1", "v1.mp4", 0, "ds", "train", original_video_id="ACTOR_01"),
            ManifestRecord("S2", "v2.mp4", 1, "ds", "test", original_video_id="ACTOR_01"),  # Leakage!
        ]
        report = ManifestManager.validate_manifest(records, check_video_paths=False)

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.leakage_violations), 1)
        self.assertIn("ACTOR_01", report.leakage_violations[0])

    def test_group_aware_splitting_guarantees_no_leakage(self):
        """Test that create_group_aware_splits ensures paired original and fake videos never separate."""
        raw_records = [
            # Group 1 (Actor 1): 1 real, 2 fakes
            ManifestRecord("R1", "v1.mp4", 0, "ds", "", original_video_id="ACTOR_01"),
            ManifestRecord("F1_A", "v2.mp4", 1, "ds", "", original_video_id="ACTOR_01"),
            ManifestRecord("F1_B", "v3.mp4", 1, "ds", "", original_video_id="ACTOR_01"),
            # Group 2 (Actor 2): 1 real, 1 fake
            ManifestRecord("R2", "v4.mp4", 0, "ds", "", original_video_id="ACTOR_02"),
            ManifestRecord("F2", "v5.mp4", 1, "ds", "", original_video_id="ACTOR_02"),
            # Group 3 (Actor 3): 1 real, 1 fake
            ManifestRecord("R3", "v6.mp4", 0, "ds", "", original_video_id="ACTOR_03"),
            ManifestRecord("F3", "v7.mp4", 1, "ds", "", original_video_id="ACTOR_03"),
            # Group 4 (Actor 4): 1 real, 1 fake
            ManifestRecord("R4", "v8.mp4", 0, "ds", "", original_video_id="ACTOR_04"),
            ManifestRecord("F4", "v9.mp4", 1, "ds", "", original_video_id="ACTOR_04"),
        ]

        partitioned = ManifestManager.create_group_aware_splits(
            raw_records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
        )

        # Validate that the resulting splits have ZERO leakage
        report = ManifestManager.validate_manifest(partitioned, check_video_paths=False)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.leakage_violations), 0)

        # Double check: ensure all records with the same original_video_id share identical split
        group_splits = {}
        for r in partitioned:
            if r.original_video_id in group_splits:
                self.assertEqual(group_splits[r.original_video_id], r.split)
            else:
                group_splits[r.original_video_id] = r.split

    def test_extended_fields_backward_compatibility(self):
        """Test round-trip of extended fields (subject_id, provenance, error_detail)."""
        records = [
            ManifestRecord(
                sample_id="EXT_001",
                video_path="videos/c23_001.mp4",
                label=0,
                dataset_name="Celeb-DF-v2",
                split="train",
                manipulation_type="original",
                original_video_id="id0_0001",
                preprocessing_status="SUCCESS",
                subject_id="id0",
                provenance="https://github.com/yuezunli/celeb-deepfakeforensic",
                error_detail="",
            )
        ]
        ManifestManager.write_manifest(records, self.csv_path)
        loaded = ManifestManager.read_manifest(self.csv_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].subject_id, "id0")
        self.assertEqual(loaded[0].provenance, "https://github.com/yuezunli/celeb-deepfakeforensic")
        self.assertEqual(loaded[0].error_detail, "")

    def test_subject_identity_leakage_detection(self):
        """Test that actor/subject identity crossing splits is flagged as a leakage violation."""
        records = [
            ManifestRecord("S1", "v1.mp4", 0, "ds", "train", subject_id="ACTOR_CELEB_01"),
            ManifestRecord("S2", "v2.mp4", 1, "ds", "val", subject_id="ACTOR_CELEB_01"),
        ]
        violations = ManifestManager.verify_leakage_safety(records)
        self.assertEqual(len(violations), 1)
        self.assertIn("ACTOR_CELEB_01", violations[0])

    def test_leakage_safe_splits_multi_level_grouping(self):
        """Test that Disjoint-Set groups both shared original video and shared subject identity safely."""
        records = [
            # Actor 1: 1 real, 1 fake sharing original, 1 fake sharing actor identity
            ManifestRecord("R1", "v1.mp4", 0, "ds", "", original_video_id="O1", subject_id="ACTOR_1"),
            ManifestRecord("F1", "v2.mp4", 1, "ds", "", original_video_id="O1", subject_id=""),
            ManifestRecord("F2", "v3.mp4", 1, "ds", "", original_video_id="O2", subject_id="ACTOR_1"),
            # Actor 2
            ManifestRecord("R2", "v4.mp4", 0, "ds", "", original_video_id="O3", subject_id="ACTOR_2"),
            ManifestRecord("F3", "v5.mp4", 1, "ds", "", original_video_id="O3", subject_id="ACTOR_2"),
            # Actor 3
            ManifestRecord("R3", "v6.mp4", 0, "ds", "", original_video_id="O4", subject_id="ACTOR_3"),
            ManifestRecord("F4", "v7.mp4", 1, "ds", "", original_video_id="O4", subject_id="ACTOR_3"),
        ]

        partitioned = ManifestManager.create_leakage_safe_splits(
            records, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=42
        )
        violations = ManifestManager.verify_leakage_safety(partitioned)
        self.assertEqual(violations, [])

        # Ensure R1, F1, and F2 (all related to Actor 1) share identical split
        actor1_splits = {r.split for r in partitioned if r.sample_id in {"R1", "F1", "F2"}}
        self.assertEqual(len(actor1_splits), 1)

    def test_deterministic_split_reproducibility(self):
        """Test that identical random seed produces identical split allocations."""
        records = [
            ManifestRecord(f"S_{i}", f"v_{i}.mp4", i % 2, "ds", "", original_video_id=f"O_{i // 2}")
            for i in range(20)
        ]
        split1 = ManifestManager.create_leakage_safe_splits(records, seed=999)
        split2 = ManifestManager.create_leakage_safe_splits(records, seed=999)

        for r1, r2 in zip(split1, split2):
            self.assertEqual(r1.sample_id, r2.sample_id)
            self.assertEqual(r1.split, r2.split)


if __name__ == "__main__":
    unittest.main()
