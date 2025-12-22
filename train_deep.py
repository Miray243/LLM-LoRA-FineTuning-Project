import os
import gc
import shutil
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

# --- Environment Setup ---
# Clear CUDA cache to prevent OOM
gc.collect()
torch.cuda.empty_cache()

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_NAME = "naholav/CodeGen-Deep-5K"
OUTPUT_DIR = "./models/deep_instruction"
SYSTEM_PROMPT = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

# LoRA Parameters
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Parameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 1024
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 100
LOGGING_STEPS = 20


# --- Custom Callback ---
class CheckpointNamingCallback(TrainerCallback):
    """
    Custom callback to rename checkpoints with step and epoch information
    for easier tracking and benchmarking.
    """

    def on_save(self, args, state, control, **kwargs):
        output_dir = args.output_dir
        # Find directories starting with 'checkpoint-'
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]

        if not checkpoints:
            return

        # Get the latest checkpoint
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        old_path = os.path.join(output_dir, latest)

        # Get current state info
        step = state.global_step
        epoch = int(state.epoch)

        # Define new structure: models/deep_instruction/checkpoints/checkpoint-step-X-epoch-Y
        new_name = f"checkpoint-step-{step}-epoch-{epoch}"
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        new_path = os.path.join(checkpoint_dir, new_name)

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Rename and move
        if os.path.exists(old_path) and not os.path.exists(new_path):
            shutil.move(old_path, new_path)
            print(f"✓ Checkpoint saved: {new_name}")


# --- Model & Tokenizer Initialization ---
print(f"Loading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
    use_cache=False
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# --- LoRA Configuration ---
print("Applying LoRA adapter...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Dataset Preparation ---
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")


def format_instruction(example):
    instruction = example.get("instruction", "")
    solution = example.get("solution", "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": solution}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


# Apply formatting and tokenization
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=os.cpu_count())

# Split dataset
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# --- Training Setup ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=10,
    load_best_model_at_end=False,
    fp16=False,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/logs",
    push_to_hub=False,
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[CheckpointNamingCallback()],
)

# --- Start Training ---
print("\nStarting training...")
trainer.train()

# --- Save Final Model ---
print("\nSaving final model...")
final_dir = os.path.join(OUTPUT_DIR, "checkpoints", "final_model")
os.makedirs(os.path.dirname(final_dir), exist_ok=True)
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

print("\nTraining complete.")
print(f"Checkpoints: {OUTPUT_DIR}/checkpoints/")
print(f"Logs: {OUTPUT_DIR}/logs/")