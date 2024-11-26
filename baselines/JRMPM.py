import torch
import torch.nn as nn


class JRMPM(nn.Module):
    def __init__(self, USER_EMBED_DIM):
        super(JRMPM, self).__init__()

        # GRU: USER_EMBED_DIM
        self.expect_sent_gru = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=USER_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=False)
        self.job_sent_gru = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=USER_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=False)

        self.expect_update_pi = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, 1, bias=False),  # 将输出降到 1 维
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-1)  # 在段落级别，仅对 batch 内的特征加权
        )

        self.job_update_pi = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, 1, bias=False),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-1)
        )

        # update g:
        self.expect_g_update = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.job_g_update = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )

        self.expect_read_phi = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, 1, bias=False),  # 将输出降到 1 维，作为 attention 权重
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-1)  # 对段落的 embedding 进行归一化
        )

        self.job_read_phi = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, 1, bias=False),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-1)
        )
        # read g:
        self.expect_g_read = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.job_g_read = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(2 * USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.ReLU(),  # 使用 ReLU 作为激活函数
            torch.nn.Dropout(0.5),  # 添加 Dropout 以防止过拟合
            torch.nn.Linear(USER_EMBED_DIM, USER_EMBED_DIM // 2),  # 第二层隐藏层
            torch.nn.ReLU(),
            torch.nn.Linear(USER_EMBED_DIM // 2, 1),  # 最后一层输出为 1
            torch.nn.Sigmoid()
        )

    def update(self, memory, a_sents, b_sents, col_mask, isexpect=True):
        # memory, a_sents, b_sents: [batch, EMBED_DIM]
        # col_mask: [batch]

        # 确保 col_mask 位于与 memory 相同的设备
        col_mask = col_mask.to(memory.device)  # 将 col_mask 移动到 memory 所在的设备

        # 计算 attention 权重 beta 和 gamma
        if isexpect:
            beta = self.expect_update_pi(memory * a_sents)  # [batch, EMBED_DIM] -> [batch, 1]
            gamma = self.expect_update_pi(memory * b_sents)  # [batch, EMBED_DIM] -> [batch, 1]
        else:
            beta = self.job_update_pi(memory * a_sents)  # [batch, EMBED_DIM] -> [batch, 1]
            gamma = self.job_update_pi(memory * b_sents)  # [batch, EMBED_DIM] -> [batch, 1]

        # 计算 i_update
        i_update = beta * a_sents + gamma * b_sents  # [batch, EMBED_DIM]

        # 计算更新门控 g_update
        if isexpect:
            g_update = self.expect_g_update(
                torch.cat([memory, i_update, memory * i_update], dim=-1))  # [batch, 3 * EMBED_DIM] -> [batch, 1]
        else:
            g_update = self.job_g_update(
                torch.cat([memory, i_update, memory * i_update], dim=-1))  # [batch, 3 * EMBED_DIM] -> [batch, 1]

        memory_update = g_update * i_update + (1 - g_update) * memory  # [batch, EMBED_DIM]

        col_mask = col_mask.unsqueeze(-1)  # [batch, 1]
        col_mask = col_mask.to(memory_update.device)  # 确保 col_mask 在与 memory_update 相同的设备上
        memory_update_mask = col_mask * memory_update  # [batch, EMBED_DIM]
        memory_noupdate_mask = (1 - col_mask) * memory  # [batch, EMBED_DIM]

        return memory_update_mask + memory_noupdate_mask  # [batch, EMBED_DIM]

    def read(self, memory, hidden_last, a_sents, isexpect=True):
        # memory: [batch, EMBED_DIM]
        # hidden_last: [batch, EMBED_DIM]
        # a_sents: [batch, EMBED_DIM]

        if isexpect:
            alpha = self.expect_read_phi(memory * hidden_last * a_sents)  # [batch, EMBED_DIM] -> [batch, 1]
        else:
            alpha = self.job_read_phi(memory * hidden_last * a_sents)  # [batch, EMBED_DIM] -> [batch, 1]

        i_read = alpha * memory  # [batch, EMBED_DIM]

        if isexpect:
            g_read = self.expect_g_read(
                torch.cat([a_sents, i_read, a_sents * i_read], dim=-1))  # [batch, 3 * EMBED_DIM] -> [batch, 1]
        else:
            g_read = self.job_g_read(
                torch.cat([a_sents, i_read, a_sents * i_read], dim=-1))  # [batch, 3 * EMBED_DIM] -> [batch, 1]

        hidden = g_read * i_read + (1 - g_read) * hidden_last  # [batch, EMBED_DIM]
        return hidden

    def process_seq(self, a_profiles, b_seq_profiless, b_seq_lens, device, isexpect=True):
        # b_seq_profiless: [batch, seq_len, EMBED_DIM]

        if isexpect:
            _, b_seq_pooled = self.expect_sent_gru(b_seq_profiless)  # [1, batch, EMBED_DIM]
        else:
            _, b_seq_pooled = self.job_sent_gru(b_seq_profiless)  # [1, batch, EMBED_DIM]
        b_seq_pooled = b_seq_pooled[-1]  # [batch, EMBED_DIM]

        seq_lengths = torch.sum(b_seq_lens, dim=1)  # [batch]
        col_mask = seq_lengths.to(device)  # [batch, 1]

        batch_memory = self.update(a_profiles, a_profiles, b_seq_pooled, col_mask, isexpect)  # [batch, EMBED_DIM]
        batch_hidden = self.read(batch_memory, a_profiles, a_profiles, isexpect)  # [batch, EMBED_DIM]

        return batch_hidden

    def predict(self, expect_hidden, job_hidden):
        combined = torch.cat([expect_hidden, job_hidden], dim=-1)  # [batch, 2 * EMBED_DIM]
        return self.MLP(combined)

    def forward(self, target_user, history_users, sequence_mask_users, target_job, history_jobs, sequence_mask_jobs,
                labels=None):
        '''
        :param target_user: [batch, EMBED_DIM]
        :param history_users: [batch, seq_len, EMBED_DIM]
        :param sequence_mask_users: [batch, seq_len]
        :param target_job: [batch, EMBED_DIM]
        :param history_jobs: [batch, seq_len, EMBED_DIM]
        :param sequence_mask_jobs: [batch, seq_len]
        :param labels: [batch]
        :return: logits: [batch]
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        job_hidden = self.process_seq(target_job, history_jobs, sequence_mask_jobs, device)  # [batch, EMBED_DIM]
        user_hidden = self.process_seq(target_user, history_users, sequence_mask_users, device)  # [batch, EMBED_DIM]
        x = self.predict(job_hidden, user_hidden).squeeze()  # [batch]

        return x
