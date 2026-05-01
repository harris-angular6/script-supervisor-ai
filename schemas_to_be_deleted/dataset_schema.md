# Dataset Schema

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
