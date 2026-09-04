import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(".")
RAW_LABELED_DIR = PROJECT_ROOT / "datasets" / "synthetic_errors"
SCENES_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched_error"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "ground_truth"

SPLITS = ["train", "val", "test"]

RAW_LABELED_FILENAMES = {
    "train": "train_error.jsonl",
    "val": "val_error.jsonl",
    "test": "test_error.jsonl",
}

SCENE_FILENAMES = {
    "train": "train_error_scenes_enriched.jsonl",
    "val": "val_error_scenes_enriched.jsonl",
    "test": "test_error_scenes_enriched.jsonl",
}

VALID_ISSUE_TYPES = {
    "prop_continuity",
    "wardrobe_continuity",
    "character_presence_continuity",
    "time_of_day_continuity",
    "location_chronology_continuity",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARNING] Failed to parse {path.name} line {line_num}: {exc}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] Wrote {len(rows)} rows to {path}")


def get_script_id(raw_record: Dict[str, Any]) -> Optional[str]:
    for key in ["script_id", "id", "source_script_id"]:
        value = raw_record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def get_annotations(raw_record: Dict[str, Any]) -> List[Any]:
    for field_name in ["error_annotations", "annotations", "labels", "issues"]:
        value = raw_record.get(field_name)
        if isinstance(value, list):
            return value
    return []


def normalize_issue_type(issue_type: Optional[str]) -> Optional[str]:
    if not issue_type:
        return None
    normalized = str(issue_type).strip().lower()
    return normalized if normalized in VALID_ISSUE_TYPES else None


def index_scenes_by_script(scene_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for row in scene_rows:
        script_id = row.get("script_id")
        scene_id = row.get("scene_id")
        scene_index = row.get("scene_index")

        if not script_id or not scene_id:
            continue

        if script_id not in grouped:
            grouped[script_id] = {
                "by_scene_id": {},
                "by_scene_index": {},
            }

        grouped[script_id]["by_scene_id"][scene_id] = row
        if isinstance(scene_index, int):
            grouped[script_id]["by_scene_index"][scene_index] = row

    return grouped


def parse_scene_index_from_value(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    text = str(value).strip()

    if text.isdigit():
        return int(text)

    match = re.search(r"scene_(\d+)$", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def resolve_scene_reference(
    script_id: str,
    ref_value: Any,
    scene_index_data: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    if script_id not in scene_index_data:
        return None

    by_scene_id = scene_index_data[script_id]["by_scene_id"]
    by_scene_index = scene_index_data[script_id]["by_scene_index"]

    if ref_value is None:
        return None

    ref_text = str(ref_value).strip()

    # full scene_id
    if ref_text in by_scene_id:
        return ref_text

    # scene_02 or numeric index
    idx = parse_scene_index_from_value(ref_value)
    if idx is not None and idx in by_scene_index:
        return by_scene_index[idx]["scene_id"]

    return None


def build_ground_truth_issue(
    gt_counter: int,
    script_id: str,
    scene_id: str,
    related_scene_id: Optional[str],
    issue_type: str,
    severity: str,
    description: str,
    evidence: List[str],
) -> Dict[str, Any]:
    return {
        "issue_id": f"GT_{script_id}_{gt_counter:04d}",
        "script_id": script_id,
        "scene_id": scene_id,
        "related_scene_id": related_scene_id,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "evidence": evidence,
        "confidence": 1.0,
    }


def convert_annotation_to_gt_row(
    script_id: str,
    ann: Any,
    scene_index_data: Dict[str, Dict[str, Any]],
    gt_counter: int,
) -> Optional[Dict[str, Any]]:
    if isinstance(ann, str):
        print(f"[DEBUG] Skipping string annotation for script {script_id}: {ann}")
        return None

    if not isinstance(ann, dict):
        print(f"[DEBUG] Skipping unsupported annotation type for script {script_id}: {type(ann)} -> {ann}")
        return None

    issue_type = normalize_issue_type(
        ann.get("issue_type") or ann.get("type") or ann.get("label")
    )
    if not issue_type:
        print(f"[DEBUG] Invalid issue_type for script {script_id}: {ann}")
        return None

    severity = str(ann.get("severity", "medium")).strip().lower()
    description = str(ann.get("description", "")).strip()

    evidence = ann.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    scene_id = None
    related_scene_id = None

    if ann.get("scene_id") is not None:
        scene_id = resolve_scene_reference(script_id, ann.get("scene_id"), scene_index_data)
        related_scene_id = resolve_scene_reference(script_id, ann.get("related_scene_id"), scene_index_data)

    elif isinstance(ann.get("scene_ids"), list):
        scene_refs = ann.get("scene_ids", [])
        if len(scene_refs) >= 2:
            related_scene_id = resolve_scene_reference(script_id, scene_refs[0], scene_index_data)
            scene_id = resolve_scene_reference(script_id, scene_refs[1], scene_index_data)
        elif len(scene_refs) == 1:
            scene_id = resolve_scene_reference(script_id, scene_refs[0], scene_index_data)

    elif ann.get("scene_index") is not None:
        scene_id = resolve_scene_reference(script_id, ann.get("scene_index"), scene_index_data)
        related_scene_id = resolve_scene_reference(script_id, ann.get("related_scene_index"), scene_index_data)

    elif ann.get("to_scene") is not None:
        related_scene_id = resolve_scene_reference(script_id, ann.get("from_scene"), scene_index_data)
        scene_id = resolve_scene_reference(script_id, ann.get("to_scene"), scene_index_data)

    if not scene_id:
        print(f"[DEBUG] Could not resolve scene for script {script_id}: {ann}")
        return None

    return build_ground_truth_issue(
        gt_counter=gt_counter,
        script_id=script_id,
        scene_id=scene_id,
        related_scene_id=related_scene_id,
        issue_type=issue_type,
        severity=severity,
        description=description,
        evidence=evidence,
    )


def process_split(split: str) -> None:
    raw_labeled_path = RAW_LABELED_DIR / RAW_LABELED_FILENAMES[split]
    scenes_path = SCENES_DIR / SCENE_FILENAMES[split]
    output_path = OUTPUT_DIR / f"{split}_ground_truth_issues.jsonl"

    raw_rows = read_jsonl(raw_labeled_path)
    scene_rows = read_jsonl(scenes_path)

    print(f"[DEBUG] Split={split}")
    print(f"[DEBUG] Loaded raw labeled rows={len(raw_rows)} from {raw_labeled_path}")
    print(f"[DEBUG] Loaded scene rows={len(scene_rows)} from {scenes_path}")

    scene_index_data = index_scenes_by_script(scene_rows)
    print(f"[DEBUG] Indexed scripts in scene data={len(scene_index_data)}")

    all_gt_rows: List[Dict[str, Any]] = []
    gt_counter = 1
    scripts_with_annotations = 0

    for raw_record in raw_rows:
        script_id = get_script_id(raw_record)
        if not script_id:
            continue

        annotations = get_annotations(raw_record)
        if not annotations:
            continue

        scripts_with_annotations += 1

        for ann in annotations:
            gt_row = convert_annotation_to_gt_row(
                script_id=script_id,
                ann=ann,
                scene_index_data=scene_index_data,
                gt_counter=gt_counter,
            )
            if gt_row is not None:
                all_gt_rows.append(gt_row)
                gt_counter += 1

    print(f"[DEBUG] scripts_with_annotations={scripts_with_annotations}")
    print(f"[DEBUG] generated_ground_truth_rows={len(all_gt_rows)}")

    write_jsonl(output_path, all_gt_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()