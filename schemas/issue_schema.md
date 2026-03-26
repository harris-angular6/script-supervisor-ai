# Issue Schema

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
