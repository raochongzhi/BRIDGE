from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

model_path = '/root/autodl-fs/model/Qwen2.5-7B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path)

llm = LLM(model=model_path,
          dtype=torch.bfloat16,
          # gpu_memory_utilization=0.8,
          # swap_space=2
         )

data = pd.read_csv('/root/autodl-fs/data/dataset.csv')

sample = '''
insert you sample
'''
sample_res = '''
insert your sample res
'''

user_his = set()
batch_size = 500  # 设置批量大小
except_list = []  # 用于记录异常的 UserID

with open('summaries_user_qwen.txt', 'w', encoding='utf-8') as file:
    file.write("UserID\tSummary_User\n")
    data1 = data.groupby('UserID', as_index=False)['resume'].first()

    # 按批次处理
    for batch_start in tqdm(range(0, len(data1), batch_size)):
        batch_end = min(batch_start + batch_size, len(data1))
        text_list = []
        user_ids = []

        # 构建当前批次的输入
        for i in range(batch_start, batch_end):
            userid = data1['UserID'][i]
            resume = data1['resume'][i]

            try:
                prompt = f"""
                根据以下用户的所有信息，撰写一段总结，概括用户的教育与职业背景、核心技能和主要成就。请确保总结简洁、全面，并能够突出用户的优势。尽量控制在150字以内，使用正式的职业语言。

                具体信息在分隔符'--------------------------'和'--------------------------'内：
                --------------------------
                {resume}
                --------------------------
                请根据以上信息生成用户的总结。
                """

                messages = [
                    {"role": "user",
                     "content": "作为一个招聘专家，你需要帮助业务负责人审核简历，快速构建简历摘要，摘要是一段连贯性的话，不要包含两段及以上的话"},
                    {"role": "assistant", "content": "当然，为了更好地理解您的任务，请给我一些简历摘要生成的案例"},
                    {"role": "user", "content": f"这里是一个例子：\n{sample}\n这是生成的结果: \n{sample_res}"},
                    {"role": "assistant",
                     "content": "好的，我已经充分理解您的需求，请给我您需要生成摘要的简历，我将为您生成满意的结果"},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                text_list.append(text)
                user_ids.append(userid)  # 保存 UserID 用于后续匹配输出

            except Exception as e:
                print(f"Error creating prompt for UserID {userid}: {e}")
                except_list.append(userid)  # 记录出错的 UserID
                continue

        # 调用批量生成接口
        try:
            sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.8,
                max_tokens=250,
                repetition_penalty=1.2
            )

            outputs = llm.generate(text_list, sampling_params)

            # 处理当前批次的输出
            for i, output in enumerate(outputs):
                try:
                    response = output.outputs[0].text.strip()
                    response = " ".join([line.strip() for line in response.splitlines() if line.strip()])  # 合并成一段话
                    file.write(f"{user_ids[i]}\t{response}\n")
                except Exception as e:
                    print(f"Error processing output for UserID {user_ids[i]}: {e}")
                    except_list.append(user_ids[i])  # 记录出错的 UserID
                    continue

        except Exception as e:
            print(f"Error generating batch from {batch_start} to {batch_end}: {e}")
            except_list.extend(user_ids)  # 记录整个批次的 UserID

# 打印异常列表
print("Exception List:", except_list)