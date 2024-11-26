import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, d_model, d_ff):
        super(DIEN, self).__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.attention_fc = nn.Linear(d_model * 2, d_ff)
        self.attention_score = nn.Linear(d_ff, 1)

    def forward(self, target, history):
        # GRU for interest evolution
        gru_out, _ = self.gru(history)
        # Attention mechanism
        target_expanded = target.unsqueeze(1).expand_as(gru_out)
        attention_input = torch.cat([gru_out, target_expanded], dim=-1)
        attention_weights = F.gelu(self.attention_fc(attention_input))
        attention_weights = self.attention_score(attention_weights).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Weighted sum of GRU outputs
        interest_representation = torch.bmm(attention_weights.unsqueeze(1), gru_out).squeeze(1)  # (batch_size, gru_hidden_size)
        # Combine target embedding with interest representation
        combined = torch.cat([interest_representation, target], dim=-1)  # (batch_size, gru_hidden_size + embed_dim)
        return combined

class DIEN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(DIEN, self).__init__()
        self.dien = DIEN(d_model, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(3584 * 3, 1024 * 4),  # 更宽的第一层
            nn.GELU(),
            nn.Linear(1024 * 4, 512 * 2),
            nn.GELU(),
            nn.Linear(512 * 2, 256),  # 输出层
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, target_user, history_users, sequence_mask_users, target_job, history_jobs, sequence_mask_jobs,
                labels=None):
        '''
        :param target_user: [bsz, d_model]
        :param history_users: [bsz, seq_len, d_model]
        :param sequence_mask_users: [bsz, seq_len]
        :param target_job: [bsz, d_model]
        :param history_jobs: [bsz, seq_len, d_model]
        :param sequence_mask_jobs: [bsz, seq_len]
        :param labels: [bsz]
        :return: logits: [bsz]
        '''
        assert not torch.isnan(target_user).any(), "target_user contains NaN"
        assert not torch.isnan(history_users).any(), "history_users contains NaN"
        assert not torch.isnan(target_job).any(), "target_job contains NaN"
        assert not torch.isnan(history_jobs).any(), "history_jobs contains NaN"

        interest_job = self.dien(target_job, history_jobs)
        concat = torch.cat([target_user, interest_job], dim=-1)
        logits = self.mlp(concat).squeeze()