from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from transformers import EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_path = '/root/autodl-tmp/model/qwen_dpo_finetuned_new'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, use_cache=False)

if tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="/root/autodl-tmp/results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=20,
    num_train_epochs=3,
    save_steps=200,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    # eval_steps=200,
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    logging_dir="/root/tf-logs",
)


def process_func(example):
    MAX_LENGTH = 384  # Qwen tokenizer may produce more tokens for a given input, so MAX_LENGTH allows for sufficient context.
    input_ids, attention_mask, labels = [], [], []

    # Tokenize the instruction and input text
    instruction = tokenizer(
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|><|im_start|>assistant\n",
        add_special_tokens=False
    )

    # Tokenize the response
    response = tokenizer(
        f"{example['output']}<|im_end|>",
        add_special_tokens=False
    )

    # Combine the input and response token IDs
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [
        1]  # Include attention for padding token
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # Truncate sequences that exceed MAX_LENGTH
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

df_train = pd.read_json('/root/autodl-fs/dataset/lora_data/tasks_train.json')
ds_train = Dataset.from_pandas(df_train)

df_eval = pd.read_json('/root/autodl-fs/dataset/lora_data/tasks_val.json')
ds_eval = Dataset.from_pandas(df_eval)

train_datasets = ds_train.map(process_func, remove_columns=ds_train.column_names)
eval_datasets = ds_eval.map(process_func, remove_columns=ds_eval.column_names)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=eval_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # callbacks=[early_stopping_callback],
)

peft_model_path="/root/autodl-tmp/model/qwen_lora_new"
trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)