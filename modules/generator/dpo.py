import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

model_path = '/root/autodl-fs/model/Qwen2.5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, use_cache=False)

if tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

model.enable_input_require_grads()
dataset = load_dataset("json", data_files="/root/autodl-fs/dataset/dpo_data/dpo_dataset.json")

dataset = dataset["train"].train_test_split(test_size=0.2, seed=729)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

training_args = DPOConfig(
    output_dir="/root/autodl-tmp/results_dpo",
    # run_name="qwen_dpo_finetuned_run",
    eval_strategy="epoch",
    save_steps=200,
    save_on_each_node=True,
    # save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    bf16=True,
    logging_dir="/root/tf-logs",
    logging_steps=10,
    report_to="tensorboard",
    # save_total_limit=2,
    # load_best_model_at_end=True,
)

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    beta=0.1,
)

trainer.train()

trainer.model.save_pretrained("/root/autodl-tmp/model/qwen_dpo_finetuned_new")
tokenizer.save_pretrained("/root/autodl-tmp/model/qwen_dpo_finetuned_new")
