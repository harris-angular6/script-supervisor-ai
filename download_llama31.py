from huggingface_hub import snapshot_download

model_name = "meta-llama/Llama-3.1-8B-Instruct"

#local_path = "./model/base_llm/llama-3.1-8b-instruct"
local_path = "./models/base_llm/llama-3.1-8b-instruct"

snapshot_download(
    repo_id=model_name,
    local_dir=local_path,
    local_dir_use_symlinks=False
)

print("Llama 3.1 8B Instruct downloaded successfully")
print("Saved at:", local_path)
