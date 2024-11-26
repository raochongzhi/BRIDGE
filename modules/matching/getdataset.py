import pandas as pd
import torch
import ast
from tqdm import tqdm

def getDataset(data_path, user_tensor_path, job_tensor_path):
    datas = pd.read_csv(data_path, encoding='utf-8') # , dtype = {'UserID': str}
    datas['UserHistory'] = datas['UserHistory'].apply(ast.literal_eval)
    datas['JobHistory'] = datas['JobHistory'].apply(ast.literal_eval)

    user_tensor_dict = torch.load(user_tensor_path, map_location='cpu')
    job_tensor_dict = torch.load(job_tensor_path, map_location='cpu')
    # print(user_tensor_dict['6925764983636627456'])
    data_list = []
    for idx, row in tqdm(datas.iterrows()):
        tmp = {}
        tmp["UserID"] = row['UserID']
        tmp["JobID"] = row['JobID']
        tmp["target_user"] = user_tensor_dict[row['UserID']]
        tmp["target_job"] = job_tensor_dict[row['JobID']]
        tmp["history_users"] = [user_tensor_dict[user] for user in row['UserHistory']]
        tmp["history_jobs"] = [job_tensor_dict[job] for job in row['JobHistory']]
        tmp["label"] = row['label']
        data_list.append(tmp)
    return data_list

