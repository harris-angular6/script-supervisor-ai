import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(".")
INPUT_DIR = PROJECT_ROOT / "datasets" / "synthetic_errors"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_error"

INPUT_FILES = ["train_error.jsonl", "val_error.jsonl", "test_error.jsonl"]


SLUGLINE_RE = re.compile(
    r"^\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.)\s+.+",
    re.IGNORECASE
)

TRANSITION_RE = re.compile(
    r"^\s*(CUT TO:|FADE IN:|FADE OUT\.?|DISSOLVE TO:|SMASH CUT TO:)\s*$",
    re.IGNORECASE
)

CHARACTER_RE = re.compile(r"^[A-Z][A-Z0-9 .'\-()]+$")

PROP_CANDIDATE_WORDS = {
    "phone", "cell phone", "smartphone", "mug", "coffee mug", "cup", "laptop",
    "computer", "bag", "purse", "briefcase", "letter", "letters", "resume",
    "resumes", "paper", "papers", "script", "screenplay", "book", "glasses",
    "hat", "coat", "jacket", "key", "keys", "wallet", "bottle", "chair",
    "desk", "table", "camera", "scarf"
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


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def is_slugline(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if TRANSITION_RE.match(line):
        return False
    return bool(SLUGLINE_RE.match(line))


def is_transition(line: str) -> bool:
    return bool(TRANSITION_RE.match(line.strip()))


def is_parenthetical(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("(") and stripped.endswith(")")


def is_character_name(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False
    if is_slugline(stripped):
        return False
    if is_transition(stripped):
        return False
    if len(stripped) > 35:
        return False
    if not CHARACTER_RE.match(stripped):
        return False

    blacklist = {
        "FADE IN", "FADE OUT", "THE END", "CUT TO", "DISSOLVE TO", "SMASH CUT TO",
        "INT", "EXT", "DAY", "NIGHT", "MORNING", "EVENING", "CONTINUOUS", "LATER"
    }
    normalized = stripped.replace(":", "").strip()
    if normalized in blacklist:
        return False

    return True


def split_into_scenes(script_text: str) -> List[Tuple[str, List[str]]]:
    lines = normalize_text(script_text).split("\n")

    scenes: List[Tuple[str, List[str]]] = []
    current_slugline: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if is_transition(line):
            continue

        if is_slugline(line):
            if current_slugline is not None:
                scenes.append((current_slugline, current_lines))
            current_slugline = line
            current_lines = []
        else:
            if current_slugline is not None:
                current_lines.append(raw_line)

    if current_slugline is not None:
        scenes.append((current_slugline, current_lines))

    return scenes


def parse_slugline(slugline: str) -> Tuple[Optional[str], Optional[str]]:
    s = slugline.strip()
    s = re.sub(r"^\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.)\s*", "", s, flags=re.IGNORECASE)

    parts = [part.strip() for part in s.split(" - ") if part.strip()]
    if not parts:
        return None, None

    time_markers = {
        "DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON",
        "DAWN", "DUSK", "LATER", "CONTINUOUS", "SAME TIME", "SUNSET"
    }

    time_of_day = None
    if parts and parts[-1].upper() in time_markers:
        time_of_day = parts[-1].upper()
        location_parts = parts[:-1]
    else:
        location_parts = parts

    location = " - ".join(location_parts).strip() if location_parts else None
    return location, time_of_day


def extract_characters_and_actions(scene_lines: List[str]) -> Tuple[List[str], List[str]]:
    characters: List[str] = []
    actions: List[str] = []

    i = 0
    while i < len(scene_lines):
        line = scene_lines[i].strip()

        if not line:
            i += 1
            continue

        if is_character_name(line):
            if line not in characters:
                characters.append(line)

            j = i + 1
            if j < len(scene_lines) and is_parenthetical(scene_lines[j].strip()):
                j += 1

            while j < len(scene_lines):
                dialogue_line = scene_lines[j].strip()
                if not dialogue_line:
                    break
                if is_slugline(dialogue_line) or is_character_name(dialogue_line):
                    break
                if is_parenthetical(dialogue_line):
                    j += 1
                    continue
                j += 1

            i = j
            continue

        if not is_parenthetical(line):
            actions.append(line)

        i += 1

    return sorted(set(characters)), actions


def extract_props_from_text(text: str) -> List[str]:
    text_lower = text.lower()
    found: List[str] = []

    for prop in sorted(PROP_CANDIDATE_WORDS):
        pattern = r"\b" + re.escape(prop.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(prop)

    normalized: List[str] = []
    seen = set()

    synonym_map = {
        "cell phone": "phone",
        "smartphone": "phone",
        "cup": "mug",
        "coffee mug": "mug",
        "letters": "letter",
        "resumes": "resume",
        "papers": "paper",
        "keys": "key",
    }

    for item in found:
        canonical = synonym_map.get(item, item)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)

    return normalized


def build_scene_record(
    script_id: str,
    scene_index: int,
    slugline: str,
    scene_lines: List[str]
) -> Dict[str, Any]:
    location, time_of_day = parse_slugline(slugline)

    clean_scene_lines = [line.strip() for line in scene_lines if line.strip()]
    scene_text = "\n".join(clean_scene_lines)

    characters, actions = extract_characters_and_actions(scene_lines)
    props = extract_props_from_text(scene_text)

    scene_id = f"{script_id}_scene_{scene_index:02d}"

    return {
        "script_id": script_id,
        "scene_id": scene_id,
        "scene_index": scene_index,
        "slugline": slugline,
        "location": location,
        "time_of_day": time_of_day,
        "scene_text": scene_text,
        "characters": characters,
        "props": props,
        "actions": actions,
    }


def parse_script_to_scenes(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    script_id = record.get("script_id", "").strip()
    script_text = record.get("script_text", "")

    if not script_id or not script_text:
        return []

    raw_scenes = split_into_scenes(script_text)

    parsed_scenes: List[Dict[str, Any]] = []
    for idx, (slugline, scene_lines) in enumerate(raw_scenes, start=1):
        parsed_scenes.append(
            build_scene_record(
                script_id=script_id,
                scene_index=idx,
                slugline=slugline,
                scene_lines=scene_lines,
            )
        )

    return parsed_scenes


def process_split_file(input_path: Path, output_path: Path) -> None:
    script_rows = read_jsonl(input_path)
    all_scenes: List[Dict[str, Any]] = []

    for record in script_rows:
        scenes = parse_script_to_scenes(record)
        all_scenes.extend(scenes)

    write_jsonl(output_path, all_scenes)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        output_name = filename.replace(".jsonl", "_scenes.jsonl")
        output_path = OUTPUT_DIR / output_name

        if not input_path.exists():
            print(f"[WARNING] Missing input file: {input_path}")
            continue

        process_split_file(input_path, output_path)


if __name__ == "__main__":
    main()