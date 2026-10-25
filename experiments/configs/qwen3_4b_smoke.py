# Quick end-to-end check of the sweep pipeline on one GPU: all methods, K=3,
# a few short samples per dataset.
target_name_or_path = "Qwen/Qwen3-4B"

methods = [
    {"name": "ar", "draft_name_or_path": None},
    {"name": "eagle3", "draft_name_or_path": "deepseek-ai/eagle3_qwen3_4b_ttt7"},
    {"name": "dflash", "draft_name_or_path": "deepseek-ai/dflash_qwen3_4b_block7"},
    {"name": "dspark", "draft_name_or_path": "deepseek-ai/dspark_qwen3_4b_block7"},
]

datasets = [
    ("gsm8k", 3),
    ("math500", 3),
    ("aime25", 3),
]

draft_lengths = [3]

generation = {
    "temperature": 1.0,
    "max_new_tokens": 128,
    "seed": 980406,
}

warmup_samples = 1
step_timing = True
output_root = "experiments/results"
