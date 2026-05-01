import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(".")
EVAL_DIR = PROJECT_ROOT / "reports" / "evaluation_error"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "analysis_error"

SPLITS = ["train", "val", "test"]

ISSUE_TYPES = [
    "prop_continuity",
    "wardrobe_continuity",
    "character_presence_continuity",
    "time_of_day_continuity",
    "location_chronology_continuity",
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote JSON: {path}")


def summarize_issue_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        issue_type = row.get("issue_type", "unknown")
        counter[issue_type] += 1

    return {issue_type: counter.get(issue_type, 0) for issue_type in ISSUE_TYPES}


def summarize_confidence(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        issue_type = row.get("issue_type", "unknown")
        confidence = row.get("confidence")
        if isinstance(confidence, (int, float)):
            grouped[issue_type].append(float(confidence))

    result: Dict[str, float] = {}
    for issue_type in ISSUE_TYPES:
        vals = grouped.get(issue_type, [])
        result[issue_type] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return result


def collect_common_evidence_phrases(rows: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, List[str]]:
    by_type: Dict[str, Counter] = {issue_type: Counter() for issue_type in ISSUE_TYPES}

    for row in rows:
        issue_type = row.get("issue_type")
        if issue_type not in by_type:
            continue

        evidence = row.get("evidence", [])
        if not isinstance(evidence, list):
            continue

        for item in evidence:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                by_type[issue_type][cleaned] += 1

    result: Dict[str, List[str]] = {}
    for issue_type in ISSUE_TYPES:
        result[issue_type] = [phrase for phrase, _ in by_type[issue_type].most_common(top_k)]

    return result


def generate_tuning_suggestions(
    fp_counts: Dict[str, int],
    fn_counts: Dict[str, int],
    fp_conf: Dict[str, float],
    fn_conf: Dict[str, float],
) -> Dict[str, List[str]]:
    suggestions: Dict[str, List[str]] = {issue_type: [] for issue_type in ISSUE_TYPES}

    for issue_type in ISSUE_TYPES:
        fp = fp_counts.get(issue_type, 0)
        fn = fn_counts.get(issue_type, 0)
        avg_fp_conf = fp_conf.get(issue_type, 0.0)
        avg_fn_conf = fn_conf.get(issue_type, 0.0)

        if fp > fn:
            suggestions[issue_type].append(
                "Detector may be too aggressive. Tighten heuristics or require stronger continuity links."
            )

        if fn > fp:
            suggestions[issue_type].append(
                "Detector may be too conservative. Expand matching logic or improve extraction coverage."
            )

        if avg_fp_conf >= 0.7 and fp > 0:
            suggestions[issue_type].append(
                "High-confidence false positives detected. Review confidence calibration and evidence rules."
            )

        if issue_type == "prop_continuity":
            suggestions[issue_type].append(
                "Inspect prop normalization and portable-object filtering."
            )

        elif issue_type == "wardrobe_continuity":
            suggestions[issue_type].append(
                "Inspect wardrobe extraction noise and require stronger character-specific evidence."
            )

        elif issue_type == "character_presence_continuity":
            suggestions[issue_type].append(
                "Inspect character extraction and avoid over-flagging group-scene transitions."
            )

        elif issue_type == "time_of_day_continuity":
            suggestions[issue_type].append(
                "Review time-jump threshold and handling of LATER / SAME TIME / CONTINUOUS markers."
            )

        elif issue_type == "location_chronology_continuity":
            suggestions[issue_type].append(
                "Inspect location parsing and whether geography changes are being over-penalized."
            )

    return suggestions


def build_markdown_report(split: str, summary: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append(f"# Error Analysis Report - {split}")
    lines.append("")
    lines.append("## False Positive Counts by Issue Type")
    lines.append("")
    lines.append("| Issue Type | Count | Avg Confidence |")
    lines.append("|---|---:|---:|")

    for issue_type in ISSUE_TYPES:
        lines.append(
            f"| {issue_type} | {summary['false_positive_counts'][issue_type]} | "
            f"{summary['false_positive_avg_confidence'][issue_type]} |"
        )

    lines.append("")
    lines.append("## False Negative Counts by Issue Type")
    lines.append("")
    lines.append("| Issue Type | Count | Avg Confidence |")
    lines.append("|---|---:|---:|")

    for issue_type in ISSUE_TYPES:
        lines.append(
            f"| {issue_type} | {summary['false_negative_counts'][issue_type]} | "
            f"{summary['false_negative_avg_confidence'][issue_type]} |"
        )

    lines.append("")
    lines.append("## Common False Positive Evidence Patterns")
    lines.append("")
    for issue_type in ISSUE_TYPES:
        lines.append(f"### {issue_type}")
        patterns = summary["false_positive_common_evidence"][issue_type]
        if not patterns:
            lines.append("- None")
        else:
            for p in patterns[:5]:
                lines.append(f"- {p}")
        lines.append("")

    lines.append("## Common False Negative Evidence Patterns")
    lines.append("")
    for issue_type in ISSUE_TYPES:
        lines.append(f"### {issue_type}")
        patterns = summary["false_negative_common_evidence"][issue_type]
        if not patterns:
            lines.append("- None")
        else:
            for p in patterns[:5]:
                lines.append(f"- {p}")
        lines.append("")

    lines.append("## Suggested Improvement Priorities")
    lines.append("")
    for issue_type in ISSUE_TYPES:
        lines.append(f"### {issue_type}")
        for s in summary["tuning_suggestions"][issue_type]:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


def analyze_split(split: str) -> Dict[str, Any]:
    fp_path = EVAL_DIR / f"{split}_false_positives.jsonl"
    fn_path = EVAL_DIR / f"{split}_false_negatives.jsonl"

    fp_rows = read_jsonl(fp_path)
    fn_rows = read_jsonl(fn_path)

    fp_counts = summarize_issue_counts(fp_rows)
    fn_counts = summarize_issue_counts(fn_rows)

    fp_conf = summarize_confidence(fp_rows)
    fn_conf = summarize_confidence(fn_rows)

    fp_evidence = collect_common_evidence_phrases(fp_rows)
    fn_evidence = collect_common_evidence_phrases(fn_rows)

    tuning_suggestions = generate_tuning_suggestions(fp_counts, fn_counts, fp_conf, fn_conf)

    return {
        "false_positive_counts": fp_counts,
        "false_negative_counts": fn_counts,
        "false_positive_avg_confidence": fp_conf,
        "false_negative_avg_confidence": fn_conf,
        "false_positive_common_evidence": fp_evidence,
        "false_negative_common_evidence": fn_evidence,
        "tuning_suggestions": tuning_suggestions,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[str, Any] = {}

    for split in SPLITS:
        summary = analyze_split(split)
        all_summaries[split] = summary

        json_path = OUTPUT_DIR / f"{split}_error_analysis_summary.json"
        md_path = OUTPUT_DIR / f"{split}_error_analysis_report.md"

        write_json(json_path, summary)
        md_path.write_text(build_markdown_report(split, summary), encoding="utf-8")
        print(f"[DONE] Wrote Markdown: {md_path}")

    write_json(OUTPUT_DIR / "all_splits_error_analysis.json", all_summaries)


if __name__ == "__main__":
    main()