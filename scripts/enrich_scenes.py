import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Uncomment this line in scripts
#from src.extraction.scene_enricher import main

PROJECT_ROOT = Path(".")
INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_error"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched_error"

INPUT_FILES = [
    "train_error_scenes.jsonl",
    "val_error_scenes.jsonl",
    "test_error_scenes.jsonl",
]


WARDROBE_COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "brown",
    "gray", "grey", "silver", "gold", "pink", "purple", "orange", "beige"
}

WARDROBE_ITEMS = {
    "coat", "jacket", "tie", "hat", "cap", "glasses", "sunglasses",
    "shirt", "dress", "sweater", "hoodie", "scarf", "boots", "shoes",
    "uniform", "blazer", "skirt", "jeans"
}

PROP_NORMALIZATION = {
    "cell phone": "phone",
    "smartphone": "phone",
    "mobile phone": "phone",
    "coffee mug": "mug",
    "cup": "mug",
    "papers": "paper",
    "letters": "letter",
    "resumes": "resume",
    "keys": "key",
    "laptop computer": "laptop",
}

CHARACTER_CLEANUP_PATTERNS = [
    r"\s*\(O\.S\.\)\s*$",
    r"\s*\(V\.O\.\)\s*$",
    r"\s*\(CONT'D\)\s*$",
    r"\s*\(OC\)\s*$",
    r"\s*\(OFF\)\s*$",
    r"'S VOICE$",
]


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


def normalize_character_name(name: str) -> str:
    cleaned = name.strip().upper()

    for pattern in CHARACTER_CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_prop(prop: str) -> str:
    p = prop.strip().lower()
    return PROP_NORMALIZATION.get(p, p)


def normalize_prop_list(props: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()

    for prop in props:
        p = normalize_prop(prop)
        if p and p not in seen:
            normalized.append(p)
            seen.add(p)

    return normalized


def extract_wardrobe(scene_text: str, characters: List[str]) -> Dict[str, List[str]]:
    wardrobe_map: Dict[str, List[str]] = {char: [] for char in characters}

    sentences = re.split(r"(?<=[.!?])\s+|\n+", scene_text)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        found_items: List[str] = []

        for color in WARDROBE_COLORS:
            for item in WARDROBE_ITEMS:
                phrase = f"{color} {item}"
                if phrase in sentence_lower:
                    found_items.append(phrase)

        for item in WARDROBE_ITEMS:
            pattern = r"\b" + re.escape(item) + r"\b"
            if re.search(pattern, sentence_lower):
                if item not in found_items:
                    found_items.append(item)

        if not found_items:
            continue

        assigned = False
        for char in characters:
            if char.lower() in sentence_lower:
                for item in found_items:
                    if item not in wardrobe_map[char]:
                        wardrobe_map[char].append(item)
                assigned = True

        if not assigned:
            continue

    return wardrobe_map


def summarize_actions(actions: List[str], max_sentences: int = 2) -> str:
    if not actions:
        return ""

    trimmed = [a.strip() for a in actions if a.strip()]
    if not trimmed:
        return ""

    summary = " ".join(trimmed[:max_sentences])
    summary = re.sub(r"\s+", " ", summary).strip()
    return summary


def compute_continuity_links(
    current_scene: Dict[str, Any],
    previous_scene: Optional[Dict[str, Any]],
    next_scene: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    previous_scene_id = previous_scene["scene_id"] if previous_scene else None
    next_scene_id = next_scene["scene_id"] if next_scene else None

    same_location_as_previous = False
    same_time_of_day_as_previous = False
    continuous_with_previous = False

    if previous_scene:
        curr_loc = (current_scene.get("location") or "").strip().upper()
        prev_loc = (previous_scene.get("location") or "").strip().upper()

        curr_time = (current_scene.get("time_of_day") or "").strip().upper()
        prev_time = (previous_scene.get("time_of_day") or "").strip().upper()

        same_location_as_previous = bool(curr_loc and prev_loc and curr_loc == prev_loc)
        same_time_of_day_as_previous = bool(curr_time and prev_time and curr_time == prev_time)

        if curr_time == "CONTINUOUS":
            continuous_with_previous = True
        elif same_location_as_previous and same_time_of_day_as_previous:
            continuous_with_previous = True
        elif same_location_as_previous and curr_time in {"LATER", "SAME TIME"}:
            continuous_with_previous = True

    return {
        "previous_scene_id": previous_scene_id,
        "next_scene_id": next_scene_id,
        "same_location_as_previous": same_location_as_previous,
        "same_time_of_day_as_previous": same_time_of_day_as_previous,
        "continuous_with_previous": continuous_with_previous,
    }


def enrich_scene_record(
    current_scene: Dict[str, Any],
    previous_scene: Optional[Dict[str, Any]],
    next_scene: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    characters = current_scene.get("characters", [])
    props = current_scene.get("props", [])
    actions = current_scene.get("actions", [])
    scene_text = current_scene.get("scene_text", "")

    normalized_characters = sorted(
        {normalize_character_name(c) for c in characters if c and c.strip()}
    )
    normalized_props = normalize_prop_list(props)
    wardrobe = extract_wardrobe(scene_text, normalized_characters)
    action_summary = summarize_actions(actions)
    continuity_links = compute_continuity_links(current_scene, previous_scene, next_scene)

    enriched = dict(current_scene)
    enriched["characters"] = normalized_characters
    enriched["normalized_props"] = normalized_props
    enriched["wardrobe"] = wardrobe
    enriched["action_summary"] = action_summary
    enriched["continuity_links"] = continuity_links

    return enriched


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


def process_scene_file(input_path: Path, output_path: Path) -> None:
    scene_rows = read_jsonl(input_path)
    grouped = group_by_script(scene_rows)

    enriched_rows: List[Dict[str, Any]] = []

    for _, scenes in grouped.items():
        for i, scene in enumerate(scenes):
            previous_scene = scenes[i - 1] if i > 0 else None
            next_scene = scenes[i + 1] if i < len(scenes) - 1 else None

            enriched_rows.append(
                enrich_scene_record(
                    current_scene=scene,
                    previous_scene=previous_scene,
                    next_scene=next_scene,
                )
            )

    write_jsonl(output_path, enriched_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        output_name = filename.replace(".jsonl", "_enriched.jsonl")
        output_path = OUTPUT_DIR / output_name

        if not input_path.exists():
            print(f"[WARNING] Missing input file: {input_path}")
            continue

        process_scene_file(input_path, output_path)


if __name__ == "__main__":
    main()