# Full comparison matrix from EXPERIMENTS.md: Qwen3-4B target, AR baseline plus
# the released DeepSpec drafters, K in {4, 7, 8, 12, 16}. 100 samples per dataset
# (all 30 for AIME25).
target_name_or_path = "Qwen/Qwen3-4B"

methods = [
    {"name": "ar", "draft_name_or_path": None},
    {"name": "eagle3", "draft_name_or_path": "deepseek-ai/eagle3_qwen3_4b_ttt7"},
    {"name": "dflash", "draft_name_or_path": "deepseek-ai/dflash_qwen3_4b_block7"},
    {"name": "dspark", "draft_name_or_path": "deepseek-ai/dspark_qwen3_4b_block7"},
]

datasets = [
    ("gsm8k", 100),
    ("math500", 100),
    ("aime25", 30),
]

draft_lengths = [4, 7, 8, 12, 16]

generation = {
    "temperature": 1.0,
    "max_new_tokens": 2048,
    "seed": 980406,
}

warmup_samples = 3
step_timing = True
output_root = "experiments/results"
