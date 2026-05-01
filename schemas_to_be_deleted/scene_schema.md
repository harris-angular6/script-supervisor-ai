# Scene Schema

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
