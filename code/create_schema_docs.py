from pathlib import Path

PROJECT_ROOT = Path(".")
SCHEMAS_DIR = PROJECT_ROOT / "schemas"

DATASET_SCHEMA_MD = """# Dataset Schema

This document describes the JSON schema for the processed script-level dataset used in the Script Supervisor AI project.

## Purpose

Each JSON object represents one full screenplay example.

Typical files:
- datasets/processed/train.jsonl
- datasets/processed/val.jsonl
- datasets/processed/test.jsonl

## Example JSON

{
  "script_id": "synthetic_script_0010",
  "dataset": "synthetic_clean",
  "source": "synthetic",
  "genre": "Drama",
  "theme": "career transition",
  "generation_prompt": "You are a professional screenwriter...",
  "script_text": "FADE IN: ...",
  "has_continuity_error": false,
  "error_annotations": []
}
"""

SCENE_SCHEMA_MD = """# Scene Schema

This schema describes scene-level data extracted from a screenplay.

Each JSON object represents a single scene.

Typical files:
- datasets/scenes/train_scenes.jsonl
- datasets/scenes/val_scenes.jsonl
- datasets/scenes/test_scenes.jsonl

## Example JSON

{
  "script_id": "synthetic_script_0010",
  "scene_id": "synthetic_script_0010_scene_01",
  "scene_index": 1,
  "slugline": "EXT. BUSY STREET - DAY",
  "location": "BUSY STREET",
  "time_of_day": "DAY",
  "scene_text": "A crowd of people scurry down the sidewalk...",
  "characters": ["JEN"],
  "props": ["phone"],
  "actions": [
    "Jen leans against a lamppost",
    "Jen checks her phone"
  ]
}
"""

ISSUE_SCHEMA_MD = """# Issue Schema

This schema describes continuity issues detected by the Script Supervisor AI system.

Each JSON object represents one detected issue.

Typical files:
- datasets/issues/predicted_issues.jsonl
- datasets/issues/labeled_issues.jsonl

## Example JSON

{
  "issue_id": "ISS_001",
  "script_id": "synthetic_script_0010",
  "scene_id": "synthetic_script_0010_scene_02",
  "related_scene_id": "synthetic_script_0010_scene_01",
  "issue_type": "prop_continuity",
  "severity": "medium",
  "description": "Jen's phone disappears between continuous scenes.",
  "evidence": [
    "Scene 01: Jen checks her phone.",
    "Scene 02: The phone is no longer present."
  ],
  "confidence": 0.83
}
"""


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Created: {path}")


def main():
    write_file(SCHEMAS_DIR / "dataset_schema.md", DATASET_SCHEMA_MD)
    write_file(SCHEMAS_DIR / "scene_schema.md", SCENE_SCHEMA_MD)
    write_file(SCHEMAS_DIR / "issue_schema.md", ISSUE_SCHEMA_MD)


if __name__ == "__main__":
    main()