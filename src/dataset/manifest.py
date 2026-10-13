"""
Dataset manifest specification, validation, and group-aware splitting for DeepGuard.

Fields specified by DeepGuard Phase 2.3:
- sample_id: Unique identifier for each video sample
- video_path: Path to the video file
- label: 0 for real/authentic, 1 for fake/manipulated
- dataset_name: Origin dataset (e.g. FaceForensics++, Celeb-DF, DFDC, synthetic)
- split: Partition assignment ('train', 'validation', 'test')
- manipulation_type: E.g. 'original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'synthetic'
- original_video_id: Source actor or original video ID when available (used for leakage-free splitting)
- preprocessing_status: Status of preprocessing ('PENDING', 'SUCCESS', 'FAILED_UNREADABLE', etc.)
"""

import csv
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


VALID_LABELS = {0, 1}
VALID_SPLITS = {"train", "validation", "test"}
SPLIT_NORMALIZATION = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}

MANIFEST_FIELDNAMES = [
    "sample_id",
    "video_path",
    "label",
    "dataset_name",
    "split",
    "manipulation_type",
    "original_video_id",
    "preprocessing_status",
]


@dataclass
class ManifestRecord:
    """Represents a single record in the DeepGuard dataset manifest."""

    sample_id: str
    video_path: str
    label: int
    dataset_name: str
    split: str
    manipulation_type: str = "unknown"
    original_video_id: str = ""
    preprocessing_status: str = "PENDING"

    def __post_init__(self):
        # Normalize split name if alias provided
        if isinstance(self.split, str):
            split_raw = self.split.strip().lower()
            self.split = SPLIT_NORMALIZATION.get(split_raw, split_raw)
        # Attempt to coerce integer label if string provided
        if isinstance(self.label, str):
            try:
                self.label = int(self.label)
            except ValueError:
                pass  # Keep raw value so validator can report exact error

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestRecord":
        """Construct record from dictionary with type coercion."""
        return cls(
            sample_id=str(data.get("sample_id", "")).strip(),
            video_path=str(data.get("video_path", "")).strip(),
            label=data.get("label", ""),
            dataset_name=str(data.get("dataset_name", "")).strip(),
            split=str(data.get("split", "")).strip(),
            manipulation_type=str(data.get("manipulation_type", "unknown")).strip(),
            original_video_id=str(data.get("original_video_id", "")).strip(),
            preprocessing_status=str(data.get("preprocessing_status", "PENDING")).strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary matching MANIFEST_FIELDNAMES."""
        return asdict(self)


@dataclass
class RecordError:
    """Error information for an invalid manifest record."""

    sample_id: str
    row_number: int
    field_name: str
    error_message: str


@dataclass
class ValidationReport:
    """Detailed summary report of manifest validation."""

    total_records: int = 0
    valid_records_count: int = 0
    invalid_records_count: int = 0
    duplicate_sample_ids: List[str] = field(default_factory=list)
    missing_video_paths: List[str] = field(default_factory=list)
    errors: List[RecordError] = field(default_factory=list)
    split_class_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "train": {"real": 0, "fake": 0, "total": 0},
            "validation": {"real": 0, "fake": 0, "total": 0},
            "test": {"real": 0, "fake": 0, "total": 0},
        }
    )
    unassigned_or_invalid_splits: Dict[str, int] = field(default_factory=dict)
    leakage_violations: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if there are zero errors, no duplicate sample IDs, and no leakage violations."""
        return (
            len(self.errors) == 0
            and len(self.duplicate_sample_ids) == 0
            and len(self.missing_video_paths) == 0
            and len(self.leakage_violations) == 0
        )

    def summary_text(self) -> str:
        """Format a human-readable summary of the validation report."""
        lines = [
            "=" * 60,
            "DEEPGUARD DATASET MANIFEST VALIDATION REPORT",
            "=" * 60,
            f"Total records inspected: {self.total_records}",
            f"Valid records:          {self.valid_records_count}",
            f"Invalid records:        {self.invalid_records_count}",
            f"Status:                 {'PASS (Clean)' if self.is_valid else 'FAIL (Issues Detected)'}",
            "-" * 60,
            "Class distribution by split:",
        ]
        for split_name, counts in self.split_class_counts.items():
            lines.append(
                f"  - {split_name.capitalize():<12}: Real (0)={counts['real']}, "
                f"Fake (1)={counts['fake']}, Total={counts['total']}"
            )
        if self.unassigned_or_invalid_splits:
            lines.append(f"  - Other/Invalid Splits: {dict(self.unassigned_or_invalid_splits)}")

        if self.duplicate_sample_ids:
            lines.append("-" * 60)
            lines.append(f"Duplicate sample_id count: {len(self.duplicate_sample_ids)}")
            lines.append(f"  Samples: {', '.join(self.duplicate_sample_ids[:10])}")

        if self.missing_video_paths:
            lines.append("-" * 60)
            lines.append(f"Missing video files on disk: {len(self.missing_video_paths)}")
            lines.append(f"  First missing: {', '.join(self.missing_video_paths[:5])}")

        if self.leakage_violations:
            lines.append("-" * 60)
            lines.append(f"Actor/Relationship Leakage Violations: {len(self.leakage_violations)}")
            for v in self.leakage_violations[:5]:
                lines.append(f"  [!] {v}")

        if self.errors:
            lines.append("-" * 60)
            lines.append(f"Field Validation Errors ({len(self.errors)} total):")
            for err in self.errors[:10]:
                lines.append(f"  Row {err.row_number} (ID: {err.sample_id}) [{err.field_name}]: {err.error_message}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ManifestManager:
    """Handles manifest reading, writing, validation, and group-aware splitting."""

    @staticmethod
    def read_manifest(csv_path: str) -> List[ManifestRecord]:
        """Read a CSV manifest file and return a list of ManifestRecord objects."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Manifest file not found: {csv_path}")

        records = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Verify required columns exist
            if reader.fieldnames is None:
                raise ValueError(f"Manifest CSV is empty: {csv_path}")
            missing_cols = set(MANIFEST_FIELDNAMES) - set(reader.fieldnames)
            if missing_cols:
                raise ValueError(f"Manifest CSV is missing required columns: {sorted(missing_cols)}")

            for row in reader:
                records.append(ManifestRecord.from_dict(row))
        return records

    @staticmethod
    def write_manifest(records: List[ManifestRecord], csv_path: str) -> None:
        """Write a list of ManifestRecord objects to a CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())

    @staticmethod
    def validate_manifest(
        records: List[ManifestRecord],
        check_video_paths: bool = False,
        base_dir: Optional[str] = None,
    ) -> ValidationReport:
        """
        Validate manifest records:
        - Check required fields are non-empty.
        - Validate label in {0, 1}.
        - Validate split in {'train', 'validation', 'test'}.
        - Detect duplicate sample IDs.
        - Optionally verify physical existence of video_path on disk.
        - Verify no related original/fake videos are placed in different splits (leakage check).
        """
        report = ValidationReport(total_records=len(records))

        seen_sample_ids: Set[str] = set()
        duplicate_ids: Set[str] = set()
        group_to_splits: Dict[str, Set[str]] = defaultdict(set)

        for idx, rec in enumerate(records, start=2):  # row 1 is header
            record_is_valid = True

            # 1. sample_id validation
            if not rec.sample_id:
                report.errors.append(
                    RecordError(rec.sample_id, idx, "sample_id", "sample_id cannot be empty")
                )
                record_is_valid = False
            elif rec.sample_id in seen_sample_ids:
                duplicate_ids.add(rec.sample_id)
                report.errors.append(
                    RecordError(rec.sample_id, idx, "sample_id", f"Duplicate sample_id: '{rec.sample_id}'")
                )
                record_is_valid = False
            else:
                seen_sample_ids.add(rec.sample_id)

            # 2. video_path validation
            if not rec.video_path:
                report.errors.append(
                    RecordError(rec.sample_id, idx, "video_path", "video_path cannot be empty")
                )
                record_is_valid = False
            elif check_video_paths:
                full_path = rec.video_path
                if base_dir and not os.path.isabs(full_path):
                    full_path = os.path.join(base_dir, full_path)
                if not os.path.exists(full_path):
                    report.missing_video_paths.append(full_path)
                    report.errors.append(
                        RecordError(rec.sample_id, idx, "video_path", f"Video file not found: '{full_path}'")
                    )
                    record_is_valid = False

            # 3. label validation
            if rec.label not in VALID_LABELS:
                report.errors.append(
                    RecordError(
                        rec.sample_id,
                        idx,
                        "label",
                        f"Invalid label '{rec.label}'. Must be 0 (real) or 1 (fake).",
                    )
                )
                record_is_valid = False

            # 4. split validation
            if rec.split not in VALID_SPLITS:
                report.errors.append(
                    RecordError(
                        rec.sample_id,
                        idx,
                        "split",
                        f"Invalid split '{rec.split}'. Must be one of: {sorted(VALID_SPLITS)}",
                    )
                )
                report.unassigned_or_invalid_splits[rec.split] = (
                    report.unassigned_or_invalid_splits.get(rec.split, 0) + 1
                )
                record_is_valid = False
            else:
                # Count split class
                if rec.label == 0:
                    report.split_class_counts[rec.split]["real"] += 1
                elif rec.label == 1:
                    report.split_class_counts[rec.split]["fake"] += 1
                report.split_class_counts[rec.split]["total"] += 1

            # 5. Track identity group for leakage prevention
            group_key = rec.original_video_id if rec.original_video_id else rec.sample_id
            if rec.split in VALID_SPLITS:
                group_to_splits[group_key].add(rec.split)

            if record_is_valid:
                report.valid_records_count += 1
            else:
                report.invalid_records_count += 1

        report.duplicate_sample_ids = sorted(list(duplicate_ids))

        # Check for group leakage across splits
        for group_id, splits in group_to_splits.items():
            if len(splits) > 1:
                violation_msg = (
                    f"Identity/relationship leakage detected: Group '{group_id}' "
                    f"is scattered across multiple splits: {sorted(list(splits))}"
                )
                report.leakage_violations.append(violation_msg)

        return report

    @staticmethod
    def create_group_aware_splits(
        records: List[ManifestRecord],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> List[ManifestRecord]:
        """
        Partition records into train, validation, and test splits without actor or
        manipulation leakage.

        If a sample has an original_video_id, all samples sharing that original_video_id
        (both authentic and manipulated derivatives) are guaranteed to be placed into
        the exact same split partition.

        Args:
            records: Input list of ManifestRecord.
            train_ratio: Proportion for training split (default 0.70).
            val_ratio: Proportion for validation split (default 0.15).
            test_ratio: Proportion for test split (default 0.15).
            seed: Random seed for deterministic reproducibility.

        Returns:
            Updated list of ManifestRecord with populated split field.
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")

        # Group records by original_video_id if provided, else sample_id
        groups: Dict[str, List[ManifestRecord]] = defaultdict(list)
        for r in records:
            group_id = r.original_video_id if r.original_video_id else r.sample_id
            groups[group_id].append(r)

        # Deterministically shuffle groups
        group_keys = sorted(list(groups.keys()))
        rng = random.Random(seed)
        rng.shuffle(group_keys)

        total_groups = len(group_keys)
        n_train = int(round(total_groups * train_ratio))
        n_val = int(round(total_groups * val_ratio))

        train_groups = set(group_keys[:n_train])
        val_groups = set(group_keys[n_train : n_train + n_val])
        test_groups = set(group_keys[n_train + n_val :])

        updated_records: List[ManifestRecord] = []
        for g_id, g_records in groups.items():
            if g_id in train_groups:
                assigned_split = "train"
            elif g_id in val_groups:
                assigned_split = "validation"
            else:
                assigned_split = "test"

            for r in g_records:
                r.split = assigned_split
                updated_records.append(r)

        return updated_records
