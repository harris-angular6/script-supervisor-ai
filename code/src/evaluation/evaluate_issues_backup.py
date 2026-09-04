import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


PROJECT_ROOT = Path(".")
PRED_DIR = PROJECT_ROOT / "datasets" / "issues"
GT_DIR = PROJECT_ROOT / "datasets" / "ground_truth"
REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"

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
    print(f"[DONE] Wrote JSON report: {path}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] Wrote JSONL report: {path}")


def issue_key(issue: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Key used for exact-match evaluation.
    """
    return (
        str(issue.get("script_id", "")),
        str(issue.get("scene_id", "")),
        str(issue.get("related_scene_id", "")),
        str(issue.get("issue_type", "")),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def index_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        indexed[issue_key(row)] = row
    return indexed


def filter_by_issue_type(rows: List[Dict[str, Any]], issue_type: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("issue_type") == issue_type]


def evaluate_split(pred_rows: List[Dict[str, Any]], gt_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_index = index_by_key(pred_rows)
    gt_index = index_by_key(gt_rows)

    pred_keys: Set[Tuple[str, str, str, str]] = set(pred_index.keys())
    gt_keys: Set[Tuple[str, str, str, str]] = set(gt_index.keys())

    tp_keys = pred_keys & gt_keys
    fp_keys = pred_keys - gt_keys
    fn_keys = gt_keys - pred_keys

    overall = {
        "tp": len(tp_keys),
        "fp": len(fp_keys),
        "fn": len(fn_keys),
    }
    overall.update(compute_metrics(overall["tp"], overall["fp"], overall["fn"]))

    per_type: Dict[str, Any] = {}
    for issue_type in ISSUE_TYPES:
        pred_type_rows = filter_by_issue_type(pred_rows, issue_type)
        gt_type_rows = filter_by_issue_type(gt_rows, issue_type)

        pred_type_keys = set(index_by_key(pred_type_rows).keys())
        gt_type_keys = set(index_by_key(gt_type_rows).keys())

        tp = len(pred_type_keys & gt_type_keys)
        fp = len(pred_type_keys - gt_type_keys)
        fn = len(gt_type_keys - pred_type_keys)

        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        metrics.update(compute_metrics(tp, fp, fn))
        per_type[issue_type] = metrics

    false_positives = [pred_index[k] for k in sorted(fp_keys)]
    false_negatives = [gt_index[k] for k in sorted(fn_keys)]
    true_positives = [pred_index[k] for k in sorted(tp_keys)]

    return {
        "overall": overall,
        "per_issue_type": per_type,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
    }


def build_markdown_report(split: str, results: Dict[str, Any]) -> str:
    overall = results["overall"]
    per_type = results["per_issue_type"]

    lines: List[str] = []
    lines.append(f"# Evaluation Report - {split}")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"- TP: {overall['tp']}")
    lines.append(f"- FP: {overall['fp']}")
    lines.append(f"- FN: {overall['fn']}")
    lines.append(f"- Precision: {overall['precision']}")
    lines.append(f"- Recall: {overall['recall']}")
    lines.append(f"- F1: {overall['f1']}")
    lines.append("")
    lines.append("## Per-Issue-Type Metrics")
    lines.append("")
    lines.append("| Issue Type | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for issue_type in ISSUE_TYPES:
        m = per_type[issue_type]
        lines.append(
            f"| {issue_type} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision']} | {m['recall']} | {m['f1']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Matching rule: exact match on script_id, scene_id, related_scene_id, and issue_type.")
    lines.append("- This report is intended for baseline rule-based detector evaluation.")
    lines.append("")

    return "\n".join(lines)


def save_reports_for_split(split: str, results: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORT_DIR / f"{split}_evaluation_summary.json"
    fp_path = REPORT_DIR / f"{split}_false_positives.jsonl"
    fn_path = REPORT_DIR / f"{split}_false_negatives.jsonl"
    tp_path = REPORT_DIR / f"{split}_true_positives.jsonl"
    md_path = REPORT_DIR / f"{split}_evaluation_report.md"

    write_json(summary_path, {
        "overall": results["overall"],
        "per_issue_type": results["per_issue_type"],
    })
    write_jsonl(fp_path, results["false_positives"])
    write_jsonl(fn_path, results["false_negatives"])
    write_jsonl(tp_path, results["true_positives"])

    markdown = build_markdown_report(split, results)
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[DONE] Wrote Markdown report: {md_path}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_split_summaries: Dict[str, Any] = {}

    for split in SPLITS:
        pred_path = PRED_DIR / f"{split}_issues.jsonl"
        gt_path = GT_DIR / f"{split}_ground_truth_issues.jsonl"

        pred_rows = read_jsonl(pred_path)
        gt_rows = read_jsonl(gt_path)

        if not gt_rows:
            print(f"[WARNING] No ground truth rows for split '{split}'. Skipping evaluation for this split.")
            continue

        results = evaluate_split(pred_rows, gt_rows)
        save_reports_for_split(split, results)

        all_split_summaries[split] = {
            "overall": results["overall"],
            "per_issue_type": results["per_issue_type"],
        }

    overall_report_path = REPORT_DIR / "all_splits_summary.json"
    write_json(overall_report_path, all_split_summaries)


if __name__ == "__main__":
    main()