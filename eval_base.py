import os
import sys
import json
import io
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from func_timeout import func_timeout, FunctionTimedOut

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "results/livecodebench"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "base_model_summary.json")
TARGET_PROBLEM_COUNT = 41
TIMEOUT_SECONDS = 3

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Loading ---
print(f"Loading model: {MODEL_ID}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Use float32 if running strictly on CPU
        device_map="auto",
        attn_implementation="sdpa"
    )
except Exception as e:
    sys.exit(f"Critical Error: Failed to load model. {e}")

# --- Dataset Preparation ---
print("Loading LiveCodeBench dataset...")
try:
    ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
except Exception as e:
    sys.exit(f"Error loading dataset: {e}")


def filter_dataset(example):
    """
    Filters the dataset for AtCoder 'Easy' problems within a specific date range.
    """
    if example['platform'] != 'atcoder':
        return False
    if example['difficulty'] != 'easy':
        return False

    # Use a broad date range initially to capture enough candidates
    contest_date = example['contest_date']
    if "2024-08-01" <= contest_date <= "2025-03-01":
        return True
    return False


filtered_ds = ds.filter(filter_dataset)

# --- Dataset Trimming ---
# We need exactly 41 problems to match the fine-tuning benchmark baseline.
current_count = len(filtered_ds)

if current_count > TARGET_PROBLEM_COUNT:
    print(f"Found {current_count} problems. Trimming to the most recent {TARGET_PROBLEM_COUNT} for consistency.")
    # Assuming dataset is chronological, select the last N problems
    start_index = current_count - TARGET_PROBLEM_COUNT
    filtered_ds = filtered_ds.select(range(start_index, current_count))
elif current_count < TARGET_PROBLEM_COUNT:
    print(f"Warning: Only found {current_count} problems. Check the date filter.")

print(f"Final evaluation set size: {len(filtered_ds)}")


# --- Execution Utilities ---
class OutputCapture(list):
    """Context manager to capture stdout during code execution."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout


def execute_code(code, input_str):
    """Executes the generated code with provided input."""
    input_lines = input_str.strip().split('\n')
    input_iterator = iter(input_lines)

    def mock_input():
        try:
            return next(input_iterator)
        except StopIteration:
            return ""

    with OutputCapture() as output:
        # Unsafe execution used for benchmarking purposes
        exec(code, {'input': mock_input, 'print': print}, {})
    return "\n".join(output).strip()


def run_test_case_safe(code, input_str):
    """Runs the test case with a strict timeout to prevent infinite loops."""
    try:
        return func_timeout(TIMEOUT_SECONDS, execute_code, args=(code, input_str))
    except FunctionTimedOut:
        return "TIMEOUT_ERROR"
    except Exception as e:
        return f"RUNTIME_ERROR: {e}"


# --- Evaluation Loop ---
passed_count = 0
print("\nStarting Benchmark...")

for example in tqdm(filtered_ds, desc="Evaluating"):
    prompt = example['question_content']

    messages = [
        {"role": "system",
         "content": "You are an expert Python programmer. Please read the problem carefully before writing any Python code. Write ONLY the executable code in a markdown block."},
        {"role": "user", "content": prompt}
    ]

    # Generate solution
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract response
    input_len = model_inputs.input_ids.shape[1]
    generated_code = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

    # Parse code block
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]

    # Run Validation
    is_passed = False
    try:
        if example['public_test_cases']:
            # Validate against the first public test case
            test_case = json.loads(example['public_test_cases'])[0]
            test_input = test_case['input']
            expected_output = test_case['output']

            result = run_test_case_safe(generated_code, test_input)

            if result != "TIMEOUT_ERROR" and result.strip() == expected_output.strip():
                is_passed = True
    except Exception:
        pass  # Fail implicitly on malformed test cases

    if is_passed:
        passed_count += 1

# --- Reporting ---
pass_rate = (passed_count / len(filtered_ds)) * 100 if len(filtered_ds) > 0 else 0

print(f"\n🏆 Base Model Evaluation Results:")
print(f"Total Problems: {len(filtered_ds)}")
print(f"Solved: {passed_count}")
print(f"Pass@1: {pass_rate:.2f}%")

# Save results
summary_data = {
    "base_model": {
        "pass@1": pass_rate / 100,
        "solved": passed_count,
        "total": len(filtered_ds)
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(summary_data, f, indent=4)

print(f"Results saved to: {OUTPUT_FILE}")