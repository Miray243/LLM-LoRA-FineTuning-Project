import os
import torch
import json
import sys
import io
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. SETTINGS ---
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_FILE = "base_model_summary.json"
os.makedirs("results", exist_ok=True)

# --- 2. LOAD MODEL AND DATASET ---
print(f"⬇️ Downloading Model: {MODEL_ID}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"  # Prevents A100/L4 Warning/Error
    )
except Exception as e:
    print(f"❌ Model loading error: {e}")
    sys.exit(1)

print("📚 Loading Dataset (AtCoder - Easy - 2408-2502)...")
# Using LiveCodeBench Lite version
ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)


# Filter: Only AtCoder, Easy difficulty, and specific date range
def filter_problems(example):
    if example['platform'] != 'atcoder': return False
    if example['difficulty'] != 'easy': return False
    date = example['contest_date']  # Format: YYYY-MM-DD
    if "2024-08" <= date <= "2025-02": return True
    return False


filtered_ds = ds.filter(filter_problems)
print(f"🎯 Total Number of Problems: {len(filtered_ds)}")


# --- 3. TEST UTILITIES ---
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout


def run_test_case(code, input_str):
    input_lines = input_str.strip().split('\n')
    input_iterator = iter(input_lines)

    def mock_input():
        try:
            return next(input_iterator)
        except StopIteration:
            return ""

    global input
    original_input = input
    input = mock_input

    output = []
    try:
        with Capturing() as output_capture:
            # Warning: exec is used here for benchmarking purposes
            exec(code, {'input': mock_input, 'print': print}, {})
        output = output_capture
    except Exception as e:
        return f"ERROR"
    finally:
        input = original_input

    return "\n".join(output).strip()


# --- 4. BENCHMARK LOOP ---
passed_count = 0

print("\n🚀 Benchmark Starting...")
for example in tqdm(filtered_ds):
    prompt_content = example['question_content']

    messages = [
        {"role": "system",
         "content": "You are an expert Python programmer. Please read the problem carefully before writing any Python code. Write ONLY the executable code in a markdown block."},
        {"role": "user", "content": prompt_content}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean Code
    code = response
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    # Execute Test (Public Test Case)
    is_passed = False
    try:
        if example['public_test_cases']:
            test_case = json.loads(example['public_test_cases'])[0]
            inp = test_case['input']
            exp = test_case['output']
            res = run_test_case(code, inp)
            if res.strip() == exp.strip():
                is_passed = True
    except:
        pass

    if is_passed:
        passed_count += 1

# --- 5. SAVE RESULTS ---
if len(filtered_ds) > 0:
    pass_rate = (passed_count / len(filtered_ds)) * 100
else:
    pass_rate = 0

print(f"\n🏆 BASE MODEL RESULTS:")
print(f"Total Problems: {len(filtered_ds)}")
print(f"Solved: {passed_count}")
print(f"Pass@1: {pass_rate:.2f}%")

summary_data = {
    "base_model": {
        "pass@1": pass_rate / 100,
        "solved": passed_count,
        "total": len(filtered_ds)
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(summary_data, f, indent=4)

print(f"✅ Results saved to '{OUTPUT_FILE}'.")