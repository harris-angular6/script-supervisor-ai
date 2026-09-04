import json
import re
from pathlib import Path
from typing import Dict, Any, Optional


RAW_DIR = Path("datasets/raw")
PROCESSED_DIR = Path("datasets/processed")


def extract_first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def split_prompt_and_script(full_text: str) -> Dict[str, str]:
    """
    Split the original synthetic text into:
    - generation_prompt
    - script_text

    We assume the screenplay begins at one of these markers:
    - FADE IN:
    - INT.
    - EXT.
    """
    screenplay_start = re.search(
        r"(?m)^(FADE IN:|INT\.|EXT\.)",
        full_text
    )

    if screenplay_start:
        start_idx = screenplay_start.start()
        generation_prompt = full_text[:start_idx].strip()
        script_text = full_text[start_idx:].strip()
    else:
        # Fallback: if no screenplay marker is found,
        # keep everything as script_text
        generation_prompt = ""
        script_text = full_text.strip()

    return {
        "generation_prompt": generation_prompt,
        "script_text": script_text,
    }


def infer_source(dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    if "synthetic" in dataset_name:
        return "synthetic"
    if "wiki" in dataset_name or "wikisource" in dataset_name:
        return "wikisource"
    return "unknown"


def infer_has_continuity_error(dataset_name: str) -> bool:
    dataset_name = dataset_name.lower()
    error_keywords = ["error", "labeled_error", "continuity_error", "corrupt", "broken"]
    return any(keyword in dataset_name for keyword in error_keywords)


def convert_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw_id = raw.get("id", "")
    dataset_name = raw.get("dataset", "unknown")
    full_text = raw.get("text", "")

    split_result = split_prompt_and_script(full_text)

    generation_prompt = split_result["generation_prompt"]
    script_text = split_result["script_text"]

    genre = extract_first_match(r"^\s*Genre:\s*(.+?)\s*$", full_text)
    theme = extract_first_match(r"^\s*Theme:\s*(.+?)\s*$", full_text)

    processed = {
        "script_id": raw_id,
        "dataset": dataset_name,
        "source": infer_source(dataset_name),
        "genre": genre,
        "theme": theme,
        "generation_prompt": generation_prompt,
        "script_text": script_text,
        "has_continuity_error": infer_has_continuity_error(dataset_name),
        "error_annotations": raw.get("error_annotations", []),
    }

    return processed


def process_jsonl_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                raw_record = json.loads(line)
                processed_record = convert_record(raw_record)
                fout.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                print(f"[WARNING] Skipping {input_path.name} line {line_num}: {exc}")

    print(f"[DONE] {input_path.name}: read={total}, written={written}, output={output_path}")


def main() -> None:
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]

    for filename in files:
        input_path = RAW_DIR / filename
        output_path = PROCESSED_DIR / filename

        if not input_path.exists():
            print(f"[WARNING] File not found: {input_path}")
            continue

        process_jsonl_file(input_path, output_path)


if __name__ == "__main__":
    main()