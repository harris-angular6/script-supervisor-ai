import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rag_continuity_detector import RagContinuityDetector

USE_RAG = True
RAG_CONFIDENCE_THRESHOLD = 0.60
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

RULE_BASED_ERROR = 0
RAG_LLM_ERROR = 1
LLM_FINETUNED_ERROR = 2

PROJECT_ROOT = Path(".")
INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched_error"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "issues_error_rag"

INPUT_FILES = [
    "train_error_scenes_enriched.jsonl",
    #"val_error_scenes_enriched.jsonl",
    #"test_error_scenes_enriched.jsonl",
]

OUTPUT_FILES_RAG_REVIEW = [
    "train_error_issues_rag_review.jsonl",
    #"val_error_issues_rag_review.jsonl",
    #"test_error_scenes_rag_review.jsonl",
]

OUTPUT_FILES = [
    "train_error_issues_rag.jsonl",
    #"val_error_issues_rag.jsonl",
    #"test_error_scenes_rag.jsonl",
]


TIME_ORDER = {
    "DAWN": 1,
    "MORNING": 2,
    "DAY": 3,
    "AFTERNOON": 4,
    "EVENING": 5,
    "DUSK": 6,
    "SUNSET": 6,
    "NIGHT": 7,
}

HIGH_VALUE_PROPS = {
    "phone", "mug", "bag", "purse", "briefcase", "letter", "resume",
    "paper", "glasses", "hat", "coat", "jacket", "key", "wallet", "bottle", "scarf"
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

def scene_link_strength(prev_scene: Dict[str, Any], curr_scene: Dict[str, Any]) -> str:
    links = curr_scene.get("continuity_links", {}) or {}

    if links.get("continuous_with_previous", False):
        return "strong"

    same_location = links.get("same_location_as_previous", False)
    same_time = links.get("same_time_of_day_as_previous", False)

    if same_location and same_time:
        return "strong"
    if same_location or same_time:
        return "medium"
    return "weak"


def make_issue(
    current_scene,
    issue_counter: int,
    script_id: str,
    scene_id: str,    
    related_scene_id: Optional[str],
    issue_type: str,
    severity: str,
    description: str,
    evidence: List[str],
    confidence: float,
) -> Dict[str, Any]:

    return {
        "continuity_issue_id": f"{script_id}_ISSUE_{issue_counter:05d}",
        "script_id": current_scene["script_id"],
        "has_continuity_error": True,        

        "detectors": {
            "rule_based": {
                "has_error": True,
                "errors": {
                        "scene_id": current_scene["scene_id"],
                        "scene_index": current_scene["scene_index"],
                        "related_scene_id": related_scene_id,
                        "issue_type": issue_type,
                        "severity": severity,
                        "description": description,
                        "evidence": evidence,
                        "confidence": round(confidence, 2),
                }                
            },
            "rag_llm": {
                "has_error": False,
                "errors": None
            },
            "fine_tuned_llm": {
                "has_error": False,
                "errors": None
            },
        },

        "final_merged_result": {
            "has_error": True,
            "issue_type": issue_type,
            "severity": severity,
            "description": description,
            "evidence": evidence,
            "confidence": round(confidence, 2),
            "supporting_detectors": [
                "rule_based_detector",
            ]
        }
    }
    
    '''
    return {
        "script_id": script_id,
        "current_scene_id": scene_id,
        "scene_index": 
        "has_continuity_error": True,
        "error_source": RULE_BASED_ERROR,
        
        "issue_id": f"{script_id}_ISS_{issue_counter:04d}",
        "script_id": script_id,
        "scene_id": scene_id,
        "related_scene_id": related_scene_id,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "evidence": evidence,
        "confidence": round(confidence, 2),
    }
    '''


def detect_prop_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    prev_props = set(prev_scene.get("normalized_props", []))
    curr_props = set(curr_scene.get("normalized_props", []))

    candidate_props = {p for p in prev_props if p in HIGH_VALUE_PROPS}
    disappeared = sorted(candidate_props - curr_props)

    if not disappeared:
        return issues, issue_counter

    if link_strength == "weak":
        return issues, issue_counter

    for prop in disappeared:
        evidence = [
            f"Previous scene props: {sorted(prev_props)}",
            f"Current scene props: {sorted(curr_props)}",
            f"Link strength between scenes: {link_strength}",
        ]

        confidence = 0.78 if link_strength == "strong" else 0.62
        severity = "medium" if link_strength == "strong" else "low"

        issues.append(
            make_issue(
                current_scene=curr_scene,
                issue_counter=issue_counter,
                script_id=curr_scene["script_id"],
                scene_id=curr_scene["scene_id"],
                related_scene_id=prev_scene["scene_id"],
                issue_type="prop_continuity",
                severity=severity,
                description=f"Prop '{prop}' appears to disappear between linked scenes without an obvious transition.",
                evidence=evidence,
                confidence=confidence,
            )
        )
        issue_counter += 1

    return issues, issue_counter


def detect_wardrobe_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    if link_strength == "weak":
        return issues, issue_counter

    prev_wardrobe = prev_scene.get("wardrobe", {}) or {}
    curr_wardrobe = curr_scene.get("wardrobe", {}) or {}

    shared_characters = sorted(set(prev_wardrobe.keys()) & set(curr_wardrobe.keys()))

    for character in shared_characters:
        prev_items = set(prev_wardrobe.get(character, []))
        curr_items = set(curr_wardrobe.get(character, []))

        if not prev_items or not curr_items:
            continue

        if prev_items != curr_items:
            evidence = [
                f"Character: {character}",
                f"Previous scene wardrobe: {sorted(prev_items)}",
                f"Current scene wardrobe: {sorted(curr_items)}",
                f"Link strength between scenes: {link_strength}",
            ]

            confidence = 0.74 if link_strength == "strong" else 0.60

            issues.append(
                make_issue(
                    current_scene=curr_scene,
                    issue_counter=issue_counter,
                    script_id=curr_scene["script_id"],
                    scene_id=curr_scene["scene_id"],
                    related_scene_id=prev_scene["scene_id"],
                    issue_type="wardrobe_continuity",
                    severity="medium",
                    description=f"Character '{character}' appears to have an inconsistent wardrobe between linked scenes.",
                    evidence=evidence,
                    confidence=confidence,
                )
            )
            issue_counter += 1

    return issues, issue_counter


def detect_character_presence_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    if link_strength == "weak":
        return issues, issue_counter

    prev_chars = set(prev_scene.get("characters", []))
    curr_chars = set(curr_scene.get("characters", []))

    disappeared = sorted(prev_chars - curr_chars)

    if not disappeared or len(disappeared) > 2:
        return issues, issue_counter

    for character in disappeared:
        evidence = [
            f"Previous scene characters: {sorted(prev_chars)}",
            f"Current scene characters: {sorted(curr_chars)}",
            f"Link strength between scenes: {link_strength}",
        ]

        confidence = 0.76 if link_strength == "strong" else 0.58
        severity = "medium" if link_strength == "strong" else "low"

        issues.append(
            make_issue(
                current_scene=curr_scene,
                issue_counter=issue_counter,
                script_id=curr_scene["script_id"],
                scene_id=curr_scene["scene_id"],
                related_scene_id=prev_scene["scene_id"],
                issue_type="character_presence_continuity",
                severity=severity,
                description=f"Character '{character}' disappears between linked scenes without a clear transition.",
                evidence=evidence,
                confidence=confidence,
            )
        )
        issue_counter += 1

    return issues, issue_counter


def detect_time_of_day_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    prev_time = (prev_scene.get("time_of_day") or "").upper().strip()
    curr_time = (curr_scene.get("time_of_day") or "").upper().strip()

    if not prev_time or not curr_time:
        return issues, issue_counter

    if prev_time == curr_time:
        return issues, issue_counter

    if curr_time in {"LATER", "SAME TIME"}:
        return issues, issue_counter

    if link_strength == "weak":
        return issues, issue_counter

    prev_rank = TIME_ORDER.get(prev_time)
    curr_rank = TIME_ORDER.get(curr_time)

    large_jump = False
    if prev_rank is not None and curr_rank is not None:
        if abs(curr_rank - prev_rank) >= 3:
            large_jump = True
    else:
        large_jump = True if link_strength == "strong" else False

    if not large_jump:
        return issues, issue_counter

    evidence = [
        f"Previous scene time_of_day: {prev_time}",
        f"Current scene time_of_day: {curr_time}",
        f"Link strength between scenes: {link_strength}",
    ]

    confidence = 0.82 if link_strength == "strong" else 0.65
    severity = "high" if link_strength == "strong" else "medium"

    issues.append(
        make_issue(
            current_scene=curr_scene,
            issue_counter=issue_counter,
            script_id=curr_scene["script_id"],
            scene_id=curr_scene["scene_id"],
            related_scene_id=prev_scene["scene_id"],
            issue_type="time_of_day_continuity",
            severity=severity,
            description="Time of day appears to shift too abruptly between linked scenes.",
            evidence=evidence,
            confidence=confidence,
        )
    )
    issue_counter += 1

    return issues, issue_counter


def detect_location_chronology_continuity(
    prev_scene: Dict[str, Any],
    curr_scene: Dict[str, Any],
    link_strength: str,
    issue_counter_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    issues: List[Dict[str, Any]] = []
    issue_counter = issue_counter_start

    prev_loc = (prev_scene.get("location") or "").upper().strip()
    curr_loc = (curr_scene.get("location") or "").upper().strip()

    prev_time = (prev_scene.get("time_of_day") or "").upper().strip()
    curr_time = (curr_scene.get("time_of_day") or "").upper().strip()

    links = curr_scene.get("continuity_links", {}) or {}
    marked_continuous = links.get("continuous_with_previous", False)

    if not prev_loc or not curr_loc:
        return issues, issue_counter

    if prev_loc == curr_loc:
        return issues, issue_counter

    should_flag = False
    confidence = 0.0
    severity = "low"

    if marked_continuous and prev_loc != curr_loc:
        should_flag = True
        confidence = 0.84
        severity = "high"
    elif link_strength == "strong" and prev_loc != curr_loc and prev_time == curr_time:
        should_flag = True
        confidence = 0.68
        severity = "medium"

    if not should_flag:
        return issues, issue_counter

    evidence = [
        f"Previous scene location: {prev_loc}",
        f"Current scene location: {curr_loc}",
        f"Previous scene time_of_day: {prev_time}",
        f"Current scene time_of_day: {curr_time}",
        f"continuous_with_previous: {marked_continuous}",
        f"Link strength between scenes: {link_strength}",
    ]

    issues.append(
        make_issue(
            current_scene=curr_scene,
            issue_counter=issue_counter,
            script_id=curr_scene["script_id"],
            scene_id=curr_scene["scene_id"],
            related_scene_id=prev_scene["scene_id"],
            issue_type="location_chronology_continuity",
            severity=severity,
            description="Scene geography or chronology appears inconsistent between linked scenes.",
            evidence=evidence,
            confidence=confidence,
        )
    )
    issue_counter += 1

    return issues, issue_counter
    
def detect_issues_for_script(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    issue_counter = 1

    for i in range(1, len(scenes)):
        prev_scene = scenes[i - 1]
        curr_scene = scenes[i]

        link_strength = scene_link_strength(prev_scene, curr_scene)

        new_issues, issue_counter = detect_prop_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_wardrobe_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_character_presence_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_time_of_day_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

        new_issues, issue_counter = detect_location_chronology_continuity(
            prev_scene, curr_scene, link_strength, issue_counter
        )
        issues.extend(new_issues)

    return issues

def detect_issues_for_script_rag_llm_review(issues: List[Dict[str, Any]], rag_collection, rag_continuity_detector) -> List[Dict[str, Any]]:

    #issues_review = List[Dict[str, Any]];
    issues_review = []
    
    for i in range(0, len(issues)):
        continuity_issue_id = issues[i]["continuity_issue_id"]

        print("\nContinuity Issue Id: ", continuity_issue_id, "\n")
        print("The issue: ", issues[i], "\n")

        issue_review_rag_llm = issue_review_result_rag_llm(continuity_issue_id, issues, rag_collection, rag_continuity_detector)

        if issue_review_rag_llm:
            print("Issue Review Rag LLM: ", issue_review_rag_llm)
            issues_review.extend(issue_review_rag_llm)

    if issues_review:
        return issues_review   
    else:
        return None

#################################################################################################################
def issue_result_rag_llm_review(continuity_issue_id, issues, rag_collection, rag_continuity_detector):

    current_scene = []
    related_scene = []

    current_scene_id = None
    related_scene_id = None
    
    #for i, issue in enumerate(issues):
    for issue in issues:
        if issue["continuity_issue_id"] == continuity_issue_id:
            current_scene_id = issue.get("detectors", {}).get("scene", {}).get("scene_id", "")
            related_sceen_id = issue.get("detectors", {}).get("related_scene", {}).get("scene_id", "")
            #current_scene_id = issue["detectors"]["rule_based"]["errors"]["scene_id"]
            #related_scene_id = issue["detectors"]["rule_based"]["errors"]["related_scene_id"]   
            
    #for scene in scenes:
    #for i in range(len(rag_collection["ids"])):

    resultCollection = rag_collection.get()

    #print("Result Collection: ", resultCollection)

    #print("Length of result Collection: ", len(resultCollection))
    #print("Length of result Collection ids: ", len(resultCollection["ids"]))
    #print("Length of resultCollection[ids][0]: ", len(resultCollection["ids"][0]))
    #print("Value of resultCollection[ids]: ", resultCollection["ids"])
    
    for i in range(len(resultCollection["ids"])):
        #print("\nContinuity Issue Id: ", continuity_issue_id)
        #print("\nResult Collection: ", resultCollection["ids"][i])
        if continuity_issue_id == resultCollection["ids"][i]:
            current_scene = {
                "script_id": resultCollection["metadatas"][i].get("scene_script_id"),
                "scene_id": resultCollection["metadatas"][i].get("scene_scene_id"),
                "scene_index": resultCollection["metadatas"][i].get("scene_scene_index"),
                "slugline": resultCollection["metadatas"][i].get("scene_slugline"),
                "location": resultCollection["metadatas"][i].get("scene_location"),
                "time_of_day": resultCollection["metadatas"][i].get("scene_time_of_day"),
                "scene_text": resultCollection["metadatas"][i].get("scene_scene_text"),
                "characters": resultCollection["metadatas"][i].get("scene_characters"),
                "props": resultCollection["metadatas"][i].get("scene_props"),
                "actions": resultCollection["metadatas"][i].get("scene_actions"),
                "normalized_props": resultCollection["metadatas"][i].get("scene_normalized_props"),
                "wardrobe": resultCollection["metadatas"][i].get("scene_wardrobe"),
                "action_summary": resultCollection["metadatas"][i].get("scene_action_summary")
            }
            #print("\nCurrent Scene: ", current_scene)
            
            related_scene = {
                "script_id": resultCollection["metadatas"][i].get("related_scene_script_id"),
                "scene_id": resultCollection["metadatas"][i].get("related_scene_id"),
                "scene_index": resultCollection["metadatas"][i].get("related_scene_index"),
                "slugline": resultCollection["metadatas"][i].get("related_scene_slugline"),
                "location": resultCollection["metadatas"][i].get("related_scene_location"),
                "time_of_day": resultCollection["metadatas"][i].get("related_scene_time_of_day"),
                "scene_text": resultCollection["metadatas"][i].get("related_scene_text"),
                "characters": resultCollection["metadatas"][i].get("related_scene_charachters"),
                "props": resultCollection["metadatas"][i].get("related_scene_props"),
                "actions": resultCollection["metadatas"][i].get("related_scene_actions"),
                "normalized_props": resultCollection["metadatas"][i].get("related_scene_normalized_props"),
                "wardrobe": resultCollection["metadatas"][i].get("related_scene_wardrobe"),
                "action_summary": resultCollection["metadatas"][i].get("related_scene_action_summary")
            }

            #print("\nRelated Scene: ", related_scene)
        
        #print("Current Scene Id: ", current_scene_id)
        #print("Scene Id in resultCollection: ", resultCollection["ids"][i])

        '''
        if current_scene_id == resultCollection["ids"][i]:
            print("Current Scene Id == resultCollection[ids][i]: ", current_scene_id, " == ", resultCollection["ids"][i])    
        
        if current_scene_id == resultCollection["ids"][i]:
            current_scene = {
                "scene_id": resultCollection["ids"][i],
                "script_id": resultCollection["metadatas"][i]["script_id"],
                "scene_index": resultCollection["metadatas"][i]["scene_index"],
                "location": resultCollection["metadatas"][i]["location"],
                "time_of_day": resultCollection["metadatas"][i]["time_of_day"],
                "scene_text": resultCollection["documents"][i]
            }
        if related_scene_id == resultCollection["ids"][i]:
            related_scene = {
                "scene_id": resultCollection["ids"][i],
                "script_id": resultCollection["metadatas"][i]["script_id"],
                "scene_index": resultCollection["metadatas"][i]["scene_index"],
                "location": resultCollection["metadatas"][i]["location"],
                "time_of_day": resultCollection["metadatas"][i]["time_of_day"],
                "scene_text": resultCollection["documents"][i]
            }    
        '''

    #print("Current scene in issue_review_result_rag_llm: ", current_scene)group_by_script

    #3print("=" * 150)
    #print("\nCurrent Scene: ", current_scene)
    #print("\nRelated Scene: ", related_scene)
    
    if current_scene and related_scene:    
        output = rag_continuity_detector.CheckContinuityWithLLM_Review(current_scene, related_scene, continuity_issue_id)
        return output
    else:
        return None

###########################################################################################################################
    
def issue_review_result_rag_llm(continuity_issue_id, issues, rag_collection, rag_continuity_detector):

    #current_scene = Dict[str, Any]
    #related_scene = Dict[str, Any]

    #print("Continuity Issue Id: ", continuity_issue_id)

    current_scene = []
    related_scene = []

    current_scene_id = None
    related_scene_id = None
    
    #for i, issue in enumerate(issues):
    for issue in issues:
        if issue["continuity_issue_id"] == continuity_issue_id:
            current_scene_id = issue["detectors"]["rule_based"]["errors"]["scene_id"]
            related_scene_id = issue["detectors"]["rule_based"]["errors"]["related_scene_id"]   
            
    #for scene in scenes:
    #for i in range(len(rag_collection["ids"])):

    resultCollection = rag_collection.get()

    #print("Result Collection: ", resultCollection)

    #print("Length of result Collection: ", len(resultCollection))
    #print("Length of result Collection ids: ", len(resultCollection["ids"]))
    #print("Length of resultCollection[ids][0]: ", len(resultCollection["ids"][0]))
    #print("Value of resultCollection[ids]: ", resultCollection["ids"])
    
    for i in range(len(resultCollection["ids"])):
        #print("Current Scene Id: ", current_scene_id)
        #print("Scene Id in resultCollection: ", resultCollection["ids"][i])

        if current_scene_id == resultCollection["ids"][i]:
            print("Current Scene Id == resultCollection[ids][i]: ", current_scene_id, " == ", resultCollection["ids"][i])    
        
        if current_scene_id == resultCollection["ids"][i]:
            current_scene = {
                "scene_id": resultCollection["ids"][i],
                "script_id": resultCollection["metadatas"][i]["script_id"],
                "scene_index": resultCollection["metadatas"][i]["scene_index"],
                "location": resultCollection["metadatas"][i]["location"],
                "time_of_day": resultCollection["metadatas"][i]["time_of_day"],
                "scene_text": resultCollection["documents"][i]
            }
        if related_scene_id == resultCollection["ids"][i]:
            related_scene = {
                "scene_id": resultCollection["ids"][i],
                "script_id": resultCollection["metadatas"][i]["script_id"],
                "scene_index": resultCollection["metadatas"][i]["scene_index"],
                "location": resultCollection["metadatas"][i]["location"],
                "time_of_day": resultCollection["metadatas"][i]["time_of_day"],
                "scene_text": resultCollection["documents"][i]
            }    

    #print("Current scene in issue_review_result_rag_llm: ", current_scene)group_by_script

    if current_scene and related_scene:    
        output = rag_continuity_detector.CheckContinuityWithLLM_Review(current_scene, related_scene, continuity_issue_id)
        return output
    else:
        return None

def process_file(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl(input_path)
    grouped = group_by_script(rows)

    all_issues: List[Dict[str, Any]] = []

    #for _, scenes in grouped.items():
    for i, (_, scenes) in enumerate(grouped.items()):
        script_issues = detect_issues_for_script(scenes)
        all_issues.extend(script_issues)

        if (i >= 10):
            break;       

    write_jsonl(output_path, all_issues)

def save_continuity_issues_with_scenes(json_inputs: list, output_file_path: str) -> None:
    with open(output_file_path, "w", encoding="utf-8") as jsonl_file:
        for item in json_inputs:
            jsonl_file.write(json.dumps(item) + "\n")


def rule_based_error_rag_llm_review(input_path: Path, output_path: Path, rag_collection, rag_continuity_detector):
    rows = read_jsonl(input_path)
    grouped = group_by_script(rows)
    
    all_issues: List[Dict[str, Any]] = []

    for i, (_, scenes) in enumerate(grouped.items()):
        script_issues_rule_based = detect_issues_for_script(scenes)

                                       
        script_issues_rag_llm_review = detect_issues_for_script_rag_llm_review(script_issues_rule_based, rag_collection, rag_continuity_detector)

        print("Script Issues: ", script_issues_rag_llm_review)
      
        if script_issues_rag_llm_review:
            all_issues.extend(script_issues_rag_llm_review)
        
        if (i >= 10):
            break;
        
    write_jsonl(output_path, all_issues)  

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
    #for i, filename in enumerate(INPUT_FILES):
        input_path = INPUT_DIR / filename
        output_name = filename.replace("_scenes_enriched.jsonl", "_issues.jsonl")
        output_path = OUTPUT_DIR / output_name

        if not input_path.exists():
            print(f"[WARNING] Missing input file: {input_path}")
            continue

        process_file(input_path, output_path)

        input_to_rag_rule = output_path
        output_path_rag_review = OUTPUT_DIR / OUTPUT_FILES_RAG_REVIEW[0] 
        
        all_rule_based_issues = read_jsonl(input_to_rag_rule)
        
        #rag_continuity_detector = RagContinuityDetector(input_path, output_path, MODEL_NAME)

        rag_review_continuity_detector_rule = RagContinuityDetector(input_to_rag_rule, output_path_rag_review, MODEL_NAME)

        all_continuity_issues = rag_review_continuity_detector_rule.LoadScenesFromRuleBasedErrorResult(scenes_info=all_rule_based_issues, jsonl_file_path=input_path)

        # print("\nAll Continuity Issues: ", all_continuity_issues, "\n")
        collection = rag_review_continuity_detector_rule.SaveContinuityIssuesToRAGDatabase(all_continuity_issues)

        #print("Collection: ", collection)
        results = collection.get()

        '''
        for i in range(len(results["ids"])):
            print("\nContinuity Issue Id: ", results["ids"][i])
            print("Documents: ", results["documents"][i])
            print("Metadata: ", results["metadatas"][i])
            print("\n", "=" * 100)
        '''

        for i in range(len(results["ids"])):
            #print("\n", "*" * 150)
            #print("\nContinuity Issue Id: ", results["ids"][i])
            #print("All Continuity_Issues: ", all_continuity_issues)
            #print("Collection: ", results)
            #print("Rag Review Continuity Detector: ", rag_review_continuity_detector_rule)                  
            
            reviewResultRAG_LLM = issue_result_rag_llm_review(results["ids"][i], all_continuity_issues, collection, rag_review_continuity_detector_rule)
            print("RAG LLM Review Result: ", reviewResultRAG_LLM)

        #rag_review_continuity_detector_rule.ClearCollection()
        # def save_continuity_issues_with_scenes(json_inputs: list, output_path: str) -> None:
        #####################################################################################
        # save_continuity_issues_with_scenes(all_continuity_issues, output_path_rag_review)
        #####################################################################################

        
        #output_path_rag = OUTPUT_DIR / OUTPUT_FILES[0]
        #rag_continuity_detector = RagContinuityDetector(input_path, output_path_rag, MODEL_NAME)

        #print("Input to rag rule: ", rag_continuity_detector.input_file_name)
        #print("\nOutput to rag rule: ", rag_continuity_detector.output_file_name)

        # input_path == "./src/dataset/scene_enriched_error/train_error_scenes_enriched.jsonl"
        #3print("Input path: ", input_path)
        
        #####################################################################
        # scenes_from_input_file = rag_continuity_detector.LoadScenes(input_to_rag_rule)
        #####################################################################

        '''
        scenes = []
        related__scenes = []

        scenes, related_scenes = rag_review_continuity_detector_rule.LoadScenesFromRuleBasedResult(scenes_info=all_rule_based_issues, jsonl_file_path=input_path)

        scenesCollection = rag_review_continuity_detector_rule.SaveSceneToRAGDatabase(scenes)

        resultsScenes = scenesCollection.get()

        print("\n", "*" * 150, "\n")
        print("Result Collection: ", scenesCollection.count())
        for i in range(len(resultsScenes["ids"])):
            print("Scene Id: ", resultsScenes["ids"][i])
            print("Document: ", resultsScenes["documents"][i])
            print("Metadata: ", resultsScenes["metadatas"][i])
            print("-" * 50)       
        
        relatedScenesCollection = rag_review_continuity_detector_rule.SaveSceneToRAGDatabase(related_scenes)

        resultsRelated = relatedScenesCollection.get()
        
        print("\n", "*" * 150, "\n")
        print("Related Collection: ", relatedScenesCollection.count())
        for i in range(len(resultsRelated["ids"])):
            print("Scene Id: ", resultsRelated["ids"][i])
            print("Document: ", resultsRelated["documents"][i])
            print("Metadata: ", resultsRelated["metadatas"][i])
            print("-" * 50)

        print("\n", "*" * 150, "\n")

        rag_review_continuity_detector_rule.ClearCollectionContent(scenesCollection)
        rag_review_continuity_detector_rule.ClearCollectionContent(relatedScenesCollection)
        '''
        
        
        #for scene in scenes_from_input_file:
        #    print("\nScene from input file: ", scene)
        
        #print("The number of scene: ", len(scenes_from_input_file))

        #for i, scene in enumerate(scenes_from_input_file):
        #    print("\nThe scene ",i, ":", scene)

        #############################################################################################
        #rag_continuity_detector.ClearCollection()
        #
        #collection = rag_continuity_detector.SaveSceneToRAGDatabase(scenes_from_input_file)
        ################################################################################################

        #print("The Number of Scenes Loaded: ", len(scenes_from_input_file))
        #def rule_based_error_rag_llm_review(input_path: Path, output_path: Path, rag_collection, rag_continuity_detector):
        
        #############################################################################################################################################################
        #rule_based_error_rag_llm_review(input_path=input_path, output_path=output_path, rag_collection=collection, rag_continuity_detector=rag_continuity_detector)
        #############################################################################################################################################################

        #for i, rule_based_issue in enumerate(all_rule_based_issues):
            # 05-20-2026 begin here

        # def BuildContinuityPromptRAGForRule(self, current_scene, previous_scene, continuity_issue_id):
        #for i, current_scene in enumerate(scenes_from_input_file):
        #    for j, rule_based_issue in enumerate(all_rule_based_issues):
        #        if current_scene["script_id"] == rule_based_issue["script_id"] and current_scene["scene_id"] == rule_based_issue["detectors"]["rule_based"]["errors"]["scene_id"]
        #        prompt = rag_continuity_detector.BuildContinuityPromptRAGForRule(current_scene, )
        
        #collection = rag_continuity_detector.LoadScenes(input_path)
        

        #def SaveSceneToRAGDatabase(self, scene, scene_document_text, metadata):
        #    self.collection.add(ids=[scene["scene_id"]], documents=[scene_document_text], metadatas=[metadata])

        
        #print("Scenes From Input File:", scenes_from_input_file)
        #print("Collection: ", collection.peek())
        
        lstResult = []
        
        #for current_scene in scenes_from_input_file:

        
        #for i , current_scene in enumerate(scenes_from_input_file):

            
            #print("Checking:", current_scene["scene_id"])

            # def GetStronglyRelatedScenes(current_scene: dict, collection, top_k: int = 10, max_distance: float = 0.8):

            #print("Current Scene: ", current_scene)
            #print("Collection: ", rag_continuity_detector.collection)
            #print("Collection: ", collection)
            
            #related_previous_scenes = rag_continuity_detector.GetStronglyRelatedScenes(current_scene, rag_continuity_detector.collection, top_k=10, max_distance=0.8)
            #related_previous_scenes = rag_continuity_detector.GetStronglyRelatedScenes(current_scene, collection, top_k=10, max_distance=0.8)

            #print("Strongly Related Scenes: ", related_previous_scenes)

            #print("Related Prev. Scenes: ", related_previous_scenes)
            # def BuildContinuityPrompt(current_scene: dict, related_scenes: list):
            #prompt = rag_continuity_detector.BuildContinuityPrompt(current_scene=current_scene, 
            #                                                       related_scenes=related_previous_scenes)

            #print("Prompt: ", prompt)

            #print(prompt)
            
            # def CheckContinuityWithLLM(self, current_scene, collection):

            #result = rag_continuity_detector.CheckContinuityWithLLM(current_scene=current_scene,
            #                                                        collection=rag_continuity_detector.collection)

            #################################################################################################
            #result = rag_continuity_detector.CheckContinuityWithLLM(current_scene=current_scene,            
            #                                                        collection=collection)
            #################################################################################################
            

            #output = self.SendPromptToModel(mistral_prompt)[0]["generated_text"]
            #if (result is not None):
            #    print("LLM returns: ", result)
            
            #lstResult.append(result)
            #print(result)
            #if i >= 20:
            #    break;

        #if i >= 10:
        #    break;
            
        #for result in lstResult:
        #    print(result)

if __name__ == "__main__":
    main()