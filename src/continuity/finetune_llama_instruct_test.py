"""
Sample QLoRA fine-tuning script: distill (rule_based + rag_llm) -> final_merged_result
into a single Llama-3.1-8B-Instruct model.
 
Assumes input JSONL records shaped like your continuity_issue records:
{
  "continuity_issue_id": ...,
  "script_id": ...,
  "has_continuity_error": bool,
  "detectors": {"rule_based": {...}, "rag_llm": {...}, "fine_tuned_llm": {...}},
  "final_merged_result": [ {...}, ... ]   # target
}
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

USE_4BIT = False

PROJECT_ROOT = Path(".")
JSON_INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes"
JSON_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "scenes"

INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched_error"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "issues_error_rag"

MODEL_ID = "./foundation_model/base_llm/llama-3.1-8b-instruct"
DATA_PATH = OUTPUT_DIR / "train_error_issues_rule_rag_merge.jsonl"
#RESPONSE_ANCHOR = "### MERGED_RESULT:"
MAX_SEQ_LEN = 2048

# ----------------------------------------------------------------------------------------
# 1. Load + flatten records into prompt/completion pairs
# ----------------------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    records = []
    skipped = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode failed ({e}) -skipping")
                skipped += 1
                continue

            if "script_id" not in record or "final_merged_result" not in record:
                print(f"Line {line_num}: missing required keys, keys present: {list(record.keys())}")
                skipped += 1
                continue
            
            records.append(json.loads(line))
    print(f"Loaded {len(records)} valid records, skipped {skipped}")
    return records

def build_prompt(record: dict) -> str:
    """Give the model the two upstream detector signals; it must reconcile them."""
    detectors = record["detectors"]
    rule_based = detectors.get("rule_based", {})
    rag_llm = detectors.get("rag_llm", {})

    user_content = {
        "You are the continuity-error merger for a file script analysis pipeline. " \
        "Given the outputs of a rule-based detector and a RAG-based LLM reviewer for " \
        "the same scene pair, reconcile them into a final merged result .Resolve " \
        "conflicts. deduplicate overlapping issues, and note which detector(s) " \
        "support each retained issue.\n\n" \
        f"RULE_BASED_OUTPUT:\n{json.dumps(rule_based, ensure_ascii=False)}\n\n" \
        f"RAG_LLM_OUTPUT:\n{json.dumps(rag_llm, ensure_ascii=False)}\n\n" \
        "Return the merged result as a JSON array."
    }
    return user_content

def build_target(record: dict) -> str:
    return json.dumps(record["final_merged_result"], ensure_ascii=False)

#def to_chat_text(tokenizer, record: dict) -> str:
def to_prompt_completion(tokenizer, record: dict) -> dict:
    prompt_content = build_prompt(record)
    target = build_target(record)

    messages = [
        {"role": "user", "content": prompt_content},
    ]
    # Render only the user turn via the chat template, then hand-append the
    # anchor + target so we control the exact literal string the collator masks on.
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    completion_text = f"{target}{tokenizer.eos_token}"
    #full_text = f"{rendered_prompt}{RESPONSE_ANCHOR}\n{target}{tokenizer.eos_token}"

    return {"prompt": rendered_prompt, "completion": completion_text}
    #return full_text
    
# -------------------------------------------------------------------------------------------
# 2. Split by script_id to prevent scene-level leakage
# -------------------------------------------------------------------------------------------

def split_by_script_id(records: list[dict], eval_frac: float = 0.1, seed: int = 42):

    #############################################################################################
    '''
    records = load_records(DATA_PATH)

    bad_records = [i for i, r in enumerate(records) if "script_id" not in r]
    print(f"{len(bad_records)} of {len(records)} records missing 'script_id'")
    print("First few bad indices:", bad_records[:10])
    
    if bad_records:
        print("Sample bad record:", records[bad_records[0]])
    '''
    #############################################################################################
    
    script_ids = sorted({r["script_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(script_ids)

    n_eval = max(1, int(len(script_ids) * eval_frac))
    eval_ids = set(script_ids[:n_eval])
    train_ids = set(script_ids[n_eval:])

    train_records = [r for r in records if r["script_id"] in train_ids]
    eval_records = [r for r in records if r["script_id"] in eval_ids]
    
    return train_records, eval_records

# ------------------------------------------------------------------------------------------
# 3. Build HF Datasets
# ------------------------------------------------------------------------------------------

def build_dataset(tokenizer, records: list[dict]) -> Dataset:
    pairs = [to_prompt_completion(tokenizer, r) for r in records]
    return Dataset.from_dict({
        "prompt": [p["prompt"] for p in pairs],
        "completion": [p["completion"] for p in pairs],
    })
    
    #texts = [to_chat_text(tokenizer, r) for r in records]
    #return Dataset.from_dict({"text": texts})

# ------------------------------------------------------------------------------------------
# 4. Verify the response-anchor token subsequence survives tokenization
#    (this is the "gotcha" step - always check before a full run)
# ------------------------------------------------------------------------------------------

def assert_anchor_tokenizable(tokenizer, sample_text: str):
    anchor_ids = tokenizer.encode(RESPONSE_ANCHOR, add_special_tokens=False)
    full_ids = tokenizer.encode(sample_text, add_special_tokens=False)

    found = any(
        full_ids[i : i + len(anchor_ids)] == anchor_ids
            for i in range(len(full_ids) - len(anchor_ids) + 1)
    )

    if not found:
        raise ValueError(
            "Response anchor token sequence not found verbatim in tokenized text. "
            "DataCollatorForCompletionOnlyLM will fall to mask correctly. "
            "Check whitespace/newlines around RESPONSE_ANCHOR."
        )

# -----------------------------------------------------------------------------------------
# 5. Model + tokenizer setup (QLoRA, NF4, sdpa for Blackwell sm_120
# -----------------------------------------------------------------------------------------

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, tokenizer

def build_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

# ----------------------------------------------------------------------------------
# 7. Training
# ----------------------------------------------------------------------------------

def main():

    print("Data Path: ", DATA_PATH)
    records = load_records(DATA_PATH)
    '''
    for i, r in enumerate(records):
        print("Record: ", r)
        if i > 5:
            break
    '''
    
    train_records, eval_records = split_by_script_id(records)
    print(f"train scripts examples={len(train_records)} eval examples={len(eval_records)}")

    model, tokenizer = load_model_and_tokenizer()

    train_dataset = build_dataset(tokenizer, train_records)
    eval_dataset = build_dataset(tokenizer, eval_records)

    print("Sample prompt:\n", train_dataset[0]["prompt"][:300])
    print("Sample completion:\n", train_dataset[0]["completion"][:300])

    # sanity check the anchor before committing to a full run
    #assert_anchor_tokenizable(tokenizer, train_database[0]["text"])

    #collator = DataCollatorForCompletionOnlyLM(
    #    response_template=RESPONSE_ANCHOR,
    #    tokenizer=tokenizer,
    #)

    sft_config = SFTConfig(
        output_dir="./merger-lora-out",
        per_device_train_batch_size=1,   # 16GB VRAM ceiling on 8B @ 4-bit
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, # effective batch size 16
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_length=MAX_SEQ_LEN,
        packing=False,                # keep False so the anchor/collator masking stays exact per-example
        completion_only_loss=True,
        report_to="none",
        #dataset_text_field="text",
        optim="paged_adamw_8bit",     # helps with DOM at this VRAM ceiling
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(),
        processing_class=tokenizer,
        #data_collator=collator,
        #tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./merger-lora-out/final")
    tokenizer.save_pretrained("./merger-lora-out/final")

if __name__ == "__main__":
    main()