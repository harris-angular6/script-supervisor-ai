import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(".")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
SCENES_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "ground_truth"

SPLITS = ["train", "val", "test"]

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


def normalize_issue_type(issue_type: Optional[str]) -> Optional[str]:
    if not issue_type:
        return None
    normalized = str(issue_type).strip().lower()
    return normalized if normalized in VALID_ISSUE_TYPES else None


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

    if ref_text in by_scene_id:
        return ref_text

    idx = parse_scene_index_from_value(ref_value)
    if idx is not None and idx in by_scene_index:
        return by_scene_index[idx]["scene_id"]

    if re.fullmatch(r"scene_\d+", ref_text, flags=re.IGNORECASE):
        idx = parse_scene_index_from_value(ref_text)
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


def annotation_to_ground_truth_rows(
    script_record: Dict[str, Any],
    scene_index_data: Dict[str, Dict[str, Any]],
    start_counter: int,
) -> Tuple[List[Dict[str, Any]], int]:
    gt_rows: List[Dict[str, Any]] = []
    gt_counter = start_counter

    script_id = script_record.get("script_id")
    annotations = script_record.get("error_annotations", [])

    if not script_id:
        print("[DEBUG] Skipping record without script_id")
        return gt_rows, gt_counter

    if not annotations:
        return gt_rows, gt_counter

    print(f"[DEBUG] Processing script_id={script_id}, annotation_count={len(annotations)}")

    for ann in annotations:
        print(f"[DEBUG] Raw annotation: {ann}")

        issue_type = normalize_issue_type(ann.get("issue_type"))
        if not issue_type:
            print(f"[DEBUG] Invalid or missing issue_type for script {script_id}: {ann.get('issue_type')}")
            continue

        severity = str(ann.get("severity", "medium")).strip().lower()
        description = str(ann.get("description", "")).strip()

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

        print(f"[DEBUG] Resolved scene_id={scene_id}, related_scene_id={related_scene_id}")

        if not scene_id:
            print(f"[DEBUG] Could not resolve scene_id for script {script_id}")
            continue

        evidence = ann.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        gt_row = build_ground_truth_issue(
            gt_counter=gt_counter,
            script_id=script_id,
            scene_id=scene_id,
            related_scene_id=related_scene_id,
            issue_type=issue_type,
            severity=severity,
            description=description,
            evidence=evidence,
        )
        gt_rows.append(gt_row)
        gt_counter += 1

    return gt_rows, gt_counter


def process_split(split: str) -> None:
    processed_path = PROCESSED_DIR / f"{split}.jsonl"
    scenes_path = SCENES_DIR / f"{split}_scenes_enriched.jsonl"
    output_path = OUTPUT_DIR / f"{split}_ground_truth_issues.jsonl"

    script_rows = read_jsonl(processed_path)
    scene_rows = read_jsonl(scenes_path)

    print(f"[DEBUG] Split={split}")
    print(f"[DEBUG] Loaded script_rows={len(script_rows)} from {processed_path}")
    print(f"[DEBUG] Loaded scene_rows={len(scene_rows)} from {scenes_path}")

    scene_index_data = index_scenes_by_script(scene_rows)
    print(f"[DEBUG] Indexed scripts in scene data: {len(scene_index_data)}")

    all_gt_rows: List[Dict[str, Any]] = []
    gt_counter = 1

    scripts_with_annotations = 0

    for script_record in script_rows:
        if script_record.get("error_annotations"):
            scripts_with_annotations += 1

        gt_rows, gt_counter = annotation_to_ground_truth_rows(
            script_record=script_record,
            scene_index_data=scene_index_data,
            start_counter=gt_counter,
        )
        all_gt_rows.extend(gt_rows)

    print(f"[DEBUG] scripts_with_annotations={scripts_with_annotations}")
    print(f"[DEBUG] generated_ground_truth_rows={len(all_gt_rows)}")

    write_jsonl(output_path, all_gt_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()