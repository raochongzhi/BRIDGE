from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
mode_path = '/root/autodl-fs/model/Qwen2.5-7B-Instruct'
lora_path = "/root/autodl-tmp/model/qwen_lora_new"

tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

model.eval()

if tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

def get_eos_embedding(text, tokenizer, model):
    inputs = tokenizer(f'{text}<|im_end|>', return_tensors="pt", padding=True, truncation=True, max_length=384)
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)
    eos_token_id = tokenizer.eos_token_id
    eos_token_index = (inputs["input_ids"] == eos_token_id) & (inputs["attention_mask"] == 1)
    eos_token_index = eos_token_index.nonzero(as_tuple=True)[1]
    last_hidden_state = outputs.hidden_states[-1]
    eos_embedding = last_hidden_state[0, eos_token_index, :]
    return eos_embedding

data_user = pd.read_csv('/root/autodl-fs/data/user.csv')
data_job = pd.read_csv('/root/autodl-fs/data/job.csv')

userid_to_embedding = {}

for _, row in tqdm(data_user.iterrows(), total=len(data_user), dynamic_ncols=True):
    user_id = row['UserID']
    summary_user = row['Summary_User']
    embedding = get_eos_embedding(summary_user, tokenizer, model)
    userid_to_embedding[user_id] = embedding.squeeze()

torch.save(userid_to_embedding, "/root/autodl-fs/dataset/embedding_data/userid_to_embedding_dpo.pt")

jobid_to_embedding = {}

for _, row in tqdm(data_job.iterrows(), total=len(data_job), dynamic_ncols=True):
    job_id = row['JobID']
    summary_job = row['Summary_Job']
    embedding = get_eos_embedding(summary_job, tokenizer, model)
    jobid_to_embedding[job_id] = embedding.squeeze()
torch.save(jobid_to_embedding, "/root/autodl-fs/dataset/embedding_data/jobid_to_embedding_dpo.pt")