import os
import re
import json
import argparse
import subprocess
import sys

# --- Configuration ---
# You can change these defaults if needed
DEFAULT_MODEL_TYPES = ["deep_instruction", "diverse_instruction"]
DEFAULT_PLATFORM = "atcoder"
DEFAULT_DIFFICULTY = "easy"


def setup_environment():
    """
    Applies necessary patches to the CodeGen repository for smooth execution.
    Specifically targets Flash Attention issues and path configurations.
    """
    print("🔧 Setting up environment and applying patches...")

    # 1. Patch common/model_loader.py to disable Flash Attention 2 requirement
    # This prevents errors on GPUs that don't support it or if it's not installed.
    loader_path = 'common/model_loader.py'
    if os.path.exists(loader_path):
        with open(loader_path, 'r') as f:
            content = f.read()

        # Replace 'flash_attention_2' with 'sdpa' (Scaled Dot Product Attention)
        if '"flash_attention_2"' in content:
            content = content.replace('"flash_attention_2"', '"sdpa"')
            with open(loader_path, 'w') as f:
                f.write(content)
            print("✅ Applied Flash Attention patch (sdpa).")
    else:
        print(f"⚠️ Warning: {loader_path} not found. Ensure you are in the CodeGen directory.")

    # 2. Patch livecodebench_eval.py for model names and dataset loading
    eval_script = 'livecodebench_eval.py'
    if os.path.exists(eval_script):
        with open(eval_script, 'r') as f:
            content = f.read()

        # Patch 1: Update model_types tuple
        # Replaces default model types with our specific project folders
        new_model_types = 'model_types: tuple = ("deep_instruction", "diverse_instruction")'
        content = re.sub(r'model_types:\s*tuple\s*=\s*\([^)]+\)', new_model_types, content)

        # Patch 2: Fix 'trust_remote_code' for datasets library
        if 'trust_remote_code=True' not in content:
            content = content.replace(
                'load_dataset(dataset_name, split=split)',
                'load_dataset(dataset_name, split=split, trust_remote_code=True)'
            )

        # Patch 3: Optimize batch size (Optional, but good for speed)
        content = re.sub(r'batch_size\s*=\s*1\b', 'batch_size=8', content)

        with open(eval_script, 'w') as f:
            f.write(content)
        print("✅ Applied patches to livecodebench_eval.py.")
    else:
        print(f"❌ Error: {eval_script} not found. Please clone the CodeGen repo first.")


def run_benchmark(model_types, platform, difficulty):
    """
    Runs the benchmark for each specified model type.
    """
    print(f"\n🚀 Starting Benchmark for: {model_types}")

    for model_type in model_types:
        print(f"\n========================================================")
        print(f"▶️ Evaluating Model: {model_type}")
        print(f"========================================================")

        # Construct command
        cmd = [
            sys.executable, "livecodebench_eval.py",
            "--model_type", model_type,
            "--platform", platform,
            "--difficulty", difficulty
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error evaluating {model_type}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run LiveCodeBench evaluation for fine-tuned models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_TYPES, help="List of model types to evaluate")
    parser.add_argument("--platform", default=DEFAULT_PLATFORM, help="Benchmark platform (e.g., atcoder)")
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY, help="Problem difficulty")

    args = parser.parse_args()

    # 1. Apply patches first
    setup_environment()

    # 2. Run benchmarks
    run_benchmark(args.models, args.platform, args.difficulty)

    print("\n✅ All benchmarks completed. Check 'results/livecodebench/summary.json' for details.")


if __name__ == "__main__":
    main()