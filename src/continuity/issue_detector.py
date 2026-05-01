import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(".")
INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched_error"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "issues_error"

INPUT_FILES = [
    "train_error_scenes_enriched.jsonl",
    "val_error_scenes_enriched.jsonl",
    "test_error_scenes_enriched.jsonl",
]


TIME_ORDER = {
    "DAWN": 1,
    "MORNING": 2,
    "DAY": 3,
    "AFTERNOON": 4,
    "EVENING": 5,
    "DUSK": 6,
    "SUNSET": 6,
    "NIGHT": 7,
}

HIGH_VALUE_PROPS = {
    "phone", "mug", "bag", "purse", "briefcase", "letter", "resume",
    "paper", "glasses", "hat", "coat", "jacket", "key", "wallet", "bottle", "scarf"
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def group_by_script(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        script_id = row.get("script_id", "")
        if not script_id:
            continue
        grouped.setdefault(script_id, []).append(row)

    for script_id in grouped:
        grouped[script_id].sort(key=lambda r: r.get("scene_index", 0))

    return grouped


def scene_link_strength(prev_scene: Dict[str, Any], curr_scene: Dict[str, Any]) -> str:
    links = curr_scene.get("continuity_links", {}) or {}

    if links.get("continuous_with_previous", False):
        return "strong"

    same_location = links.get("same_location_as_previous", False)
    same_time = links.get("same_time_of_day_as_previous", False)

    if same_location and same_time:
        return "strong"
    if same_location or same_time:
        return "medium"
    return "weak"


def make_issue(
    issue_counter: int,
    script_id: str,
    scene_id: str,
    related_scene_id: Optional[str],
    issue_type: str,
    severity: str,
    description: str,
    evidence: List[str],
    confidence: float,
) -> Dict[str, Any]:
    return {
        "issue_id": f"{script_id}_ISS_{issue_counter:04d}",
        "script_id": script_id,
        "scene_id": scene_id,
        "related_scene_id": related_scene_id,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "evidence": evidence,
        "confidence": round(confidence, 2),
    }


def detect_prop_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    prev_props = set(prev_scene.get("normalized_props", []))
    curr_props = set(curr_scene.get("normalized_props", []))

    candidate_props = {p for p in prev_props if p in HIGH_VALUE_PROPS}
    disappeared = sorted(candidate_props - curr_props)

    if not disappeared:
        return issues, issue_counter

    if link_strength == "weak":
        return issues, issue_counter

    for prop in disappeared:
        evidence = [
            f"Previous scene props: {sorted(prev_props)}",
            f"Current scene props: {sorted(curr_props)}",
            f"Link strength between scenes: {link_strength}",
        ]

        confidence = 0.78 if link_strength == "strong" else 0.62
        severity = "medium" if link_strength == "strong" else "low"

        issues.append(
            make_issue(
                issue_counter=issue_counter,
                script_id=curr_scene["script_id"],
                scene_id=curr_scene["scene_id"],
                related_scene_id=prev_scene["scene_id"],
                issue_type="prop_continuity",
                severity=severity,
                description=f"Prop '{prop}' appears to disappear between linked scenes without an obvious transition.",
                evidence=evidence,
                confidence=confidence,
            )
        )
        issue_counter += 1

    return issues, issue_counter


def detect_wardrobe_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    if link_strength == "weak":
        return issues, issue_counter

    prev_wardrobe = prev_scene.get("wardrobe", {}) or {}
    curr_wardrobe = curr_scene.get("wardrobe", {}) or {}

    shared_characters = sorted(set(prev_wardrobe.keys()) & set(curr_wardrobe.keys()))

    for character in shared_characters:
        prev_items = set(prev_wardrobe.get(character, []))
        curr_items = set(curr_wardrobe.get(character, []))

        if not prev_items or not curr_items:
            continue

        if prev_items != curr_items:
            evidence = [
                f"Character: {character}",
                f"Previous scene wardrobe: {sorted(prev_items)}",
                f"Current scene wardrobe: {sorted(curr_items)}",
                f"Link strength between scenes: {link_strength}",
            ]

            confidence = 0.74 if link_strength == "strong" else 0.60

            issues.append(
                make_issue(
                    issue_counter=issue_counter,
                    script_id=curr_scene["script_id"],
                    scene_id=curr_scene["scene_id"],
                    related_scene_id=prev_scene["scene_id"],
                    issue_type="wardrobe_continuity",
                    severity="medium",
                    description=f"Character '{character}' appears to have an inconsistent wardrobe between linked scenes.",
                    evidence=evidence,
                    confidence=confidence,
                )
            )
            issue_counter += 1

    return issues, issue_counter


def detect_character_presence_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    if link_strength == "weak":
        return issues, issue_counter

    prev_chars = set(prev_scene.get("characters", []))
    curr_chars = set(curr_scene.get("characters", []))

    disappeared = sorted(prev_chars - curr_chars)

    if not disappeared or len(disappeared) > 2:
        return issues, issue_counter

    for character in disappeared:
        evidence = [
            f"Previous scene characters: {sorted(prev_chars)}",
            f"Current scene characters: {sorted(curr_chars)}",
            f"Link strength between scenes: {link_strength}",
        ]

        confidence = 0.76 if link_strength == "strong" else 0.58
        severity = "medium" if link_strength == "strong" else "low"

        issues.append(
            make_issue(
                issue_counter=issue_counter,
                script_id=curr_scene["script_id"],
                scene_id=curr_scene["scene_id"],
                related_scene_id=prev_scene["scene_id"],
                issue_type="character_presence_continuity",
                severity=severity,
                description=f"Character '{character}' disappears between linked scenes without a clear transition.",
                evidence=evidence,
                confidence=confidence,
            )
        )
        issue_counter += 1

    return issues, issue_counter


def detect_time_of_day_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    prev_time = (prev_scene.get("time_of_day") or "").upper().strip()
    curr_time = (curr_scene.get("time_of_day") or "").upper().strip()

    if not prev_time or not curr_time:
        return issues, issue_counter

    if prev_time == curr_time:
        return issues, issue_counter

    if curr_time in {"LATER", "SAME TIME"}:
        return issues, issue_counter

    if link_strength == "weak":
        return issues, issue_counter

    prev_rank = TIME_ORDER.get(prev_time)
    curr_rank = TIME_ORDER.get(curr_time)

    large_jump = False
    if prev_rank is not None and curr_rank is not None:
        if abs(curr_rank - prev_rank) >= 3:
            large_jump = True
    else:
        large_jump = True if link_strength == "strong" else False

    if not large_jump:
        return issues, issue_counter

    evidence = [
        f"Previous scene time_of_day: {prev_time}",
        f"Current scene time_of_day: {curr_time}",
        f"Link strength between scenes: {link_strength}",
    ]

    confidence = 0.82 if link_strength == "strong" else 0.65
    severity = "high" if link_strength == "strong" else "medium"

    issues.append(
        make_issue(
            issue_counter=issue_counter,
            script_id=curr_scene["script_id"],
            scene_id=curr_scene["scene_id"],
            related_scene_id=prev_scene["scene_id"],
            issue_type="time_of_day_continuity",
            severity=severity,
            description="Time of day appears to shift too abruptly between linked scenes.",
            evidence=evidence,
            confidence=confidence,
        )
    )
    issue_counter += 1

    return issues, issue_counter


def detect_location_chronology_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    prev_loc = (prev_scene.get("location") or "").upper().strip()
    curr_loc = (curr_scene.get("location") or "").upper().strip()

    prev_time = (prev_scene.get("time_of_day") or "").upper().strip()
    curr_time = (curr_scene.get("time_of_day") or "").upper().strip()

    links = curr_scene.get("continuity_links", {}) or {}
    marked_continuous = links.get("continuous_with_previous", False)

    if not prev_loc or not curr_loc:
        return issues, issue_counter

    if prev_loc == curr_loc:
        return issues, issue_counter

    should_flag = False
    confidence = 0.0
    severity = "low"

    if marked_continuous and prev_loc != curr_loc:
        should_flag = True
        confidence = 0.84
        severity = "high"
    elif link_strength == "strong" and prev_loc != curr_loc and prev_time == curr_time:
        should_flag = True
        confidence = 0.68
        severity = "medium"

    if not should_flag:
        return issues, issue_counter

    evidence = [
        f"Previous scene location: {prev_loc}",
        f"Current scene location: {curr_loc}",
        f"Previous scene time_of_day: {prev_time}",
        f"Current scene time_of_day: {curr_time}",
        f"continuous_with_previous: {marked_continuous}",
        f"Link strength between scenes: {link_strength}",
    ]

    issues.append(
        make_issue(
            issue_counter=issue_counter,
            script_id=curr_scene["script_id"],
            scene_id=curr_scene["scene_id"],
            related_scene_id=prev_scene["scene_id"],
            issue_type="location_chronology_continuity",
            severity=severity,
            description="Scene geography or chronology appears inconsistent between linked scenes.",
            evidence=evidence,
            confidence=confidence,
        )
    )
    issue_counter += 1

    return issues, issue_counter


def detect_issues_for_script(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    issue_counter = 1

    for i in range(1, len(scenes)):
        prev_scene = scenes[i - 1]
        curr_scene = scenes[i]

        link_strength = scene_link_strength(prev_scene, curr_scene)

        new_issues, issue_counter = detect_prop_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_wardrobe_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_character_presence_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_time_of_day_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_location_chronology_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

    return issues


def process_file(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl(input_path)
    grouped = group_by_script(rows)

    all_issues: List[Dict[str, Any]] = []

    for _, scenes in grouped.items():
        script_issues = detect_issues_for_script(scenes)
        all_issues.extend(script_issues)

    write_jsonl(output_path, all_issues)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        output_name = filename.replace("_scenes_enriched.jsonl", "_issues.jsonl")
        output_path = OUTPUT_DIR / output_name

        if not input_path.exists():
            print(f"[WARNING] Missing input file: {input_path}")
            continue

        process_file(input_path, output_path)


if __name__ == "__main__":
    main()