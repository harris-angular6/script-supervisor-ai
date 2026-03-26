import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(".")
#INPUT_DIR = PROJECT_ROOT / "datasets" / "clean"
INPUT_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "synthetic_errors"

INPUT_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]

VALID_ISSUE_TYPES = {
    "prop_continuity",
    "wardrobe_continuity",
    "character_presence_continuity",
    "time_of_day_continuity",
    "location_chronology_continuity",
}

TIME_SWAP_MAP = {
    "DAY": "NIGHT",
    "NIGHT": "DAY",
    "MORNING": "NIGHT",
    "EVENING": "MORNING",
    "AFTERNOON": "NIGHT",
}

WARDROBE_ITEMS = [
    "red coat", "blue jacket", "black hat", "green scarf", "silver glasses"
]

PROPS = [
    "phone", "mug", "bag", "briefcase", "letter", "resume", "glasses", "key"
]


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


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def split_into_scenes(script_text: str) -> List[Dict[str, Any]]:
    """
    Simple scene splitter that returns:
    [
      {"scene_index": 1, "slugline": "...", "scene_text": "..."},
      ...
    ]
    """
    slugline_re = re.compile(r"^\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.)\s+.+", re.IGNORECASE)
    transition_re = re.compile(r"^\s*(FADE IN:|FADE OUT\.?|CUT TO:|DISSOLVE TO:)\s*$", re.IGNORECASE)

    text = normalize_text(script_text)
    lines = text.split("\n")

    scenes: List[Dict[str, Any]] = []
    current_slugline: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if transition_re.match(line):
            continue

        if slugline_re.match(line):
            if current_slugline is not None:
                scenes.append({
                    "slugline": current_slugline,
                    "scene_text": "\n".join(current_lines).strip()
                })
            current_slugline = line
            current_lines = []
        else:
            if current_slugline is not None:
                current_lines.append(raw_line)

    if current_slugline is not None:
        scenes.append({
            "slugline": current_slugline,
            "scene_text": "\n".join(current_lines).strip()
        })

    for i, scene in enumerate(scenes, start=1):
        scene["scene_index"] = i
        scene["scene_key"] = f"scene_{i:02d}"

    return scenes


def rebuild_script_from_scenes(scenes: List[Dict[str, Any]]) -> str:
    parts: List[str] = ["FADE IN:", ""]
    for scene in scenes:
        parts.append(scene["slugline"])
        parts.append("")
        if scene["scene_text"]:
            parts.append(scene["scene_text"])
            parts.append("")
    parts.append("FADE OUT.")
    return "\n".join(parts).strip()


def choose_adjacent_scene_pair(scenes: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if len(scenes) < 2:
        return None
    idx = random.randint(0, len(scenes) - 2)
    return idx, idx + 1


def choose_scene_with_keyword(scenes: List[Dict[str, Any]], keywords: List[str]) -> Optional[int]:
    candidates = []
    for i, scene in enumerate(scenes):
        scene_text_lower = scene["scene_text"].lower()
        if any(k.lower() in scene_text_lower for k in keywords):
            candidates.append(i)
    if not candidates:
        return None
    return random.choice(candidates)


def inject_prop_continuity(scenes: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    pair = choose_adjacent_scene_pair(scenes)
    if pair is None:
        return None

    prev_idx, curr_idx = pair
    new_scenes = deepcopy(scenes)

    prev_scene = new_scenes[prev_idx]
    curr_scene = new_scenes[curr_idx]

    target_prop = None
    for prop in PROPS:
        if prop in prev_scene["scene_text"].lower():
            target_prop = prop
            break

    if target_prop is None:
        target_prop = random.choice(PROPS)
        # Inject into previous scene if absent so the discontinuity becomes meaningful
        prev_scene["scene_text"] += f"\nA {target_prop} is clearly visible."

    pattern = re.compile(rf"\b{re.escape(target_prop)}\b", re.IGNORECASE)
    modified_text, count = pattern.subn("", curr_scene["scene_text"])

    if count == 0:
        # Force mismatch by adding a sentence that omits the prop despite continuity
        modified_text = curr_scene["scene_text"] + "\nThe object is no longer mentioned."
    modified_text = re.sub(r"\s+", " ", modified_text).strip()
    curr_scene["scene_text"] = modified_text

    ann = {
        "issue_type": "prop_continuity",
        "scene_ids": [prev_scene["scene_key"], curr_scene["scene_key"]],
        "description": f"Prop '{target_prop}' disappears between linked scenes.",
        "severity": "medium",
        "evidence": [
            f"{prev_scene['scene_key']} contains '{target_prop}'.",
            f"{curr_scene['scene_key']} was modified to remove or omit '{target_prop}'."
        ],
        "injection_method": "remove_prop_reference",
        "target_prop": target_prop,
    }
    return new_scenes, ann


def inject_time_of_day_continuity(scenes: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    pair = choose_adjacent_scene_pair(scenes)
    if pair is None:
        return None

    prev_idx, curr_idx = pair
    new_scenes = deepcopy(scenes)

    curr_scene = new_scenes[curr_idx]
    slugline = curr_scene["slugline"]

    matched = None
    for src_time, dst_time in TIME_SWAP_MAP.items():
        if re.search(rf"\b{src_time}\b", slugline, re.IGNORECASE):
            matched = (src_time, dst_time)
            break

    if matched is None:
        # Add a time marker if missing
        src_time, dst_time = "DAY", "NIGHT"
        curr_scene["slugline"] = slugline + f" - {dst_time}"
    else:
        src_time, dst_time = matched
        curr_scene["slugline"] = re.sub(rf"\b{src_time}\b", dst_time, slugline, flags=re.IGNORECASE)

    prev_scene = new_scenes[prev_idx]

    ann = {
        "issue_type": "time_of_day_continuity",
        "scene_ids": [prev_scene["scene_key"], curr_scene["scene_key"]],
        "description": f"Time of day shifts abruptly from {src_time} to {dst_time}.",
        "severity": "high",
        "evidence": [
            f"{prev_scene['scene_key']} precedes {curr_scene['scene_key']}.",
            f"{curr_scene['scene_key']} slugline was modified to use '{dst_time}'."
        ],
        "injection_method": "swap_time_of_day_marker",
        "original_time_of_day": src_time,
        "new_time_of_day": dst_time,
    }
    return new_scenes, ann


def inject_location_chronology_continuity(scenes: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    pair = choose_adjacent_scene_pair(scenes)
    if pair is None:
        return None

    prev_idx, curr_idx = pair
    new_scenes = deepcopy(scenes)

    curr_scene = new_scenes[curr_idx]
    prev_scene = new_scenes[prev_idx]

    original_slugline = curr_scene["slugline"]
    new_location = random.choice([
        "INT. AIRPORT - DAY",
        "EXT. ROOFTOP - NIGHT",
        "INT. COURTROOM - DAY",
        "EXT. BEACH - SUNSET"
    ])
    curr_scene["slugline"] = new_location

    ann = {
        "issue_type": "location_chronology_continuity",
        "scene_ids": [prev_scene["scene_key"], curr_scene["scene_key"]],
        "description": "Location shifts abruptly between adjacent scenes.",
        "severity": "high",
        "evidence": [
            f"{curr_scene['scene_key']} slugline changed from '{original_slugline}' to '{new_location}'."
        ],
        "injection_method": "swap_slugline_location",
        "original_slugline": original_slugline,
        "new_slugline": new_location,
    }
    return new_scenes, ann


def inject_wardrobe_continuity(scenes: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    pair = choose_adjacent_scene_pair(scenes)
    if pair is None:
        return None

    prev_idx, curr_idx = pair
    new_scenes = deepcopy(scenes)

    prev_scene = new_scenes[prev_idx]
    curr_scene = new_scenes[curr_idx]

    wardrobe_a = random.choice(WARDROBE_ITEMS)
    wardrobe_b = random.choice([w for w in WARDROBE_ITEMS if w != wardrobe_a])

    # Force one scene to mention wardrobe_a and the next to mention wardrobe_b
    prev_scene["scene_text"] += f"\nJEN is wearing a {wardrobe_a}."
    curr_scene["scene_text"] += f"\nJEN is now wearing a {wardrobe_b}."

    ann = {
        "issue_type": "wardrobe_continuity",
        "scene_ids": [prev_scene["scene_key"], curr_scene["scene_key"]],
        "description": f"Wardrobe changes from '{wardrobe_a}' to '{wardrobe_b}' between linked scenes.",
        "severity": "medium",
        "evidence": [
            f"{prev_scene['scene_key']} mentions '{wardrobe_a}'.",
            f"{curr_scene['scene_key']} mentions '{wardrobe_b}'."
        ],
        "injection_method": "swap_wardrobe_description",
        "target_character": "JEN",
        "original_wardrobe": wardrobe_a,
        "new_wardrobe": wardrobe_b,
    }
    return new_scenes, ann


def inject_character_presence_continuity(scenes: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    pair = choose_adjacent_scene_pair(scenes)
    if pair is None:
        return None

    prev_idx, curr_idx = pair
    new_scenes = deepcopy(scenes)

    prev_scene = new_scenes[prev_idx]
    curr_scene = new_scenes[curr_idx]

    target_character = "JEN"

    if target_character not in prev_scene["scene_text"].upper():
        prev_scene["scene_text"] += f"\n{target_character} enters and remains in the room."

    # Remove character mention from current scene if possible
    pattern = re.compile(rf"\b{re.escape(target_character)}\b", re.IGNORECASE)
    modified_text, count = pattern.subn("", curr_scene["scene_text"])

    if count == 0:
        modified_text = curr_scene["scene_text"] + "\nOnly Gordon remains present."

    modified_text = re.sub(r"\s+", " ", modified_text).strip()
    curr_scene["scene_text"] = modified_text

    ann = {
        "issue_type": "character_presence_continuity",
        "scene_ids": [prev_scene["scene_key"], curr_scene["scene_key"]],
        "description": f"Character '{target_character}' disappears between linked scenes without explanation.",
        "severity": "medium",
        "evidence": [
            f"{prev_scene['scene_key']} includes '{target_character}'.",
            f"{curr_scene['scene_key']} was modified to remove or omit '{target_character}'."
        ],
        "injection_method": "remove_character_reference",
        "target_character": target_character,
    }
    return new_scenes, ann


INJECTORS = [
    inject_prop_continuity,
    inject_wardrobe_continuity,
    inject_character_presence_continuity,
    inject_time_of_day_continuity,
    inject_location_chronology_continuity,
]


def inject_one_error(script_record: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    script_id = script_record.get("script_id")
    script_text = script_record.get("script_text", "")

    if not script_id or not script_text:
        return None

    scenes = split_into_scenes(script_text)
    if len(scenes) < 2:
        return None

    injector = rng.choice(INJECTORS)
    result = injector(scenes)
    if result is None:
        return None

    new_scenes, annotation = result
    new_script_text = rebuild_script_from_scenes(new_scenes)

    return {
        "script_id": f"{script_id}_error",
        "source_script_id": script_id,
        "dataset": "synthetic_error",
        "source": script_record.get("source", "synthetic"),
        "genre": script_record.get("genre"),
        "theme": script_record.get("theme"),
        "script_text": new_script_text,
        "has_continuity_error": True,
        "error_annotations": [annotation],
    }


def process_split(input_path: Path, output_path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    rows = read_jsonl(input_path)

    output_rows: List[Dict[str, Any]] = []
    skipped = 0

    for row in rows:
        result = inject_one_error(row, rng)
        if result is None:
            skipped += 1
            continue
        output_rows.append(result)

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Output rows: {len(output_rows)}")
    print(f"[INFO] Skipped rows: {skipped}")

    write_jsonl(output_path, output_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename.replace(".jsonl", "_error.jsonl")

        if not input_path.exists():
            print(f"[WARNING] Missing input file: {input_path}")
            continue

        process_split(input_path, output_path)


if __name__ == "__main__":
    main()