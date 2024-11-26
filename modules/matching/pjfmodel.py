import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from JRMPM import JRMPM
from shpjf import SHPJF


class CasualAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(CasualAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        assert self.d_k * num_heads == self.d_model, \
            "Embedding size must be divisible by num_heads"
        self.queries = nn.Linear(self.d_model, self.d_model)
        self.keys = nn.Linear(self.d_model, self.d_model)
        self.values = nn.Linear(self.d_model, self.d_model)
        self.output = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def casual_attention(self, query, key, value, mask, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = scores - scores.max(dim=-1, keepdim=True)[0]
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        out = torch.matmul(scores, value)
        return out

    def forward(self, history_items, sequence_mask):
        batch_size, seq_len, _ = history_items.size()

        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(history_items.device)
        if sequence_mask is not None:
            sequence_mask = sequence_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask.logical_and(sequence_mask)
        else:
            combined_mask = causal_mask

        query = self.queries(history_items).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.keys(history_items).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.values(history_items).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = self.casual_attention(query, key, value, mask=combined_mask, dropout=self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.output(concat)
        out = self.layer_norm(out + history_items)
        return out


class TargetAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TargetAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def attention(self, query, key, value, mask, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = scores - scores.max(dim=-1, keepdim=True)[0]
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        out = torch.matmul(scores, value)
        return out

    def forward(self, target_item, history_item_repr, sequence_mask):
        batch_size, seq_len, _ = history_item_repr.size()

        sequence_mask = sequence_mask.unsqueeze(1).unsqueeze(2)

        query = self.query_layer(target_item).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_layer(history_item_repr).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_layer(history_item_repr).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = self.attention(query, key, value, sequence_mask, dropout=self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.output(concat)
        out = self.layer_norm(out + target_item)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ClassificationLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model // 2, d_ff)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DIEN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(DIEN, self).__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.attention_fc = nn.Linear(d_model * 2, d_ff)
        self.attention_score = nn.Linear(d_ff, 1)
        # self.output_fc = nn.Linear(d_model * 2, 1)

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
        interest_representation = torch.bmm(attention_weights.unsqueeze(1), gru_out).squeeze(
            1)  # (batch_size, gru_hidden_size)
        # Combine target embedding with interest representation
        combined = torch.cat([interest_representation, target], dim=-1)  # (batch_size, gru_hidden_size + embed_dim)
        return combined


class UserJobMatchingModel(nn.Module):
    def __init__(self, d_model1, num_heads1, dropout1, d_model2, num_heads2, dropout2, dropout3, model_type):
        super(UserJobMatchingModel, self).__init__()
        self.causal_attention = CasualAttention(d_model1, num_heads1, dropout1)
        self.target_attention = TargetAttention(d_model2, num_heads2, dropout2)

        self.layer_norm = nn.LayerNorm(d_model1 + d_model2)

        self.mlp1 = ClassificationLayer(d_model1 + d_model2, 1, dropout3)
        self.mlp2 = ClassificationLayer(10, 1, dropout3)
        self.ffn1 = FeedForwardNetwork(d_model1 + d_model2, (d_model1 + d_model2) * 4, dropout3)

        self.mlp = nn.Sequential(
            nn.Linear(3584 * 2, 1024 * 2),  # 更宽的第一层
            nn.GELU(),
            nn.Linear(1024 * 2, 512),
            nn.GELU(),
            nn.Linear(512, 1)  # 输出层
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_type = model_type

        # for DIEN model
        self.dien = DIEN(d_model1, d_model1 // 2)
        self.mlp3 = nn.Sequential(
            nn.Linear(3584 * 3, 1024 * 4),  # 更宽的第一层
            nn.GELU(),
            nn.Linear(1024 * 4, 512 * 2),
            nn.GELU(),
            nn.Linear(512 * 2, 256),  # 输出层
            nn.GELU(),
            nn.Linear(256, 1)
        )
        ## for jrmpm model
        self.jrmpm = JRMPM(d_model2)

        ## for shpjf model
        self.shpjf = SHPJF(d_model1, 10, d_model2, d_model1)

        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.apply(initialize_weights)

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

        if self.model_type == 'BRIDGE':
            history_user_repr = self.causal_attention(history_users, sequence_mask_users)
            target_user = target_user.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_user_repr = self.target_attention(target_user, history_user_repr,
                                                     sequence_mask_jobs)  # [bsz, 1, d_model]
            history_job_repr = self.causal_attention(history_jobs, sequence_mask_jobs)
            target_job = target_job.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_job_repr = self.target_attention(target_job, history_job_repr,
                                                    sequence_mask_jobs)  # [bsz, 1, d_model]

            concat_history = torch.cat([target_user_repr, target_job_repr], dim=-1)  # [bsz, 1, d_model * 2]
            concat_target = torch.cat([target_user, target_job], dim=-1)  # [bsz, 1, d_model * 2]
            concat = self.layer_norm(concat_history + concat_target)  # [bsz, 1, d_model * 2]
            logits = self.mlp(concat).squeeze()  # [bsz]
        # below models are for ablation studies
        elif self.model_type == 'wo-history':
            target_user = target_user.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_job = target_job.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]

            concat_target = torch.cat([target_user, target_job], dim=-1)  # [bsz, 1, d_model * 2]
            logits = self.mlp(concat_target).squeeze()
        elif self.model_type == 'wo-casual':
            target_user = target_user.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_user_repr = self.target_attention(target_user, history_users,
                                                     sequence_mask_jobs)  # [bsz, 1, d_model]
            target_job = target_job.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_job_repr = self.target_attention(target_job, history_jobs, sequence_mask_jobs)  # [bsz, 1, d_model]

            concat_history = torch.cat([target_user_repr, target_job_repr], dim=-1)  # [bsz, 1, d_model * 2]
            concat_target = torch.cat([target_user, target_job], dim=-1)  # [bsz, 1, d_model * 2]
            concat = self.layer_norm(concat_history + concat_target)  # [bsz, 1, d_model * 2]
            logits = self.mlp1(concat).squeeze()  # [bsz]
        elif self.model_type == 'wo-target':
            history_user_repr = self.causal_attention(history_users, sequence_mask_users)  # [bsz, 10, d_model]
            target_user = target_user.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_user_repr = self.mlp2(history_user_repr.permute(0, 2, 1)).permute(0, 2, 1)

            history_job_repr = self.causal_attention(history_jobs, sequence_mask_jobs)
            target_job = target_job.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_job_repr = self.mlp2(history_job_repr.permute(0, 2, 1)).permute(0, 2, 1)

            concat_history = torch.cat([target_user_repr, target_job_repr], dim=-1)  # [bsz, 1, d_model * 2]
            concat_target = torch.cat([target_user, target_job], dim=-1)  # [bsz, 1, d_model * 2]
            concat = self.layer_norm(concat_history + concat_target)  # [bsz, 1, d_model * 2]
            logits = self.mlp1(concat).squeeze()  # [bsz]
        elif self.model_type == 'wo-attention':
            # history_user_repr = self.causal_attention(history_users, sequence_mask_users) # [bsz, 10, d_model]
            target_user = target_user.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_user_repr = self.mlp2(history_users.permute(0, 2, 1)).permute(0, 2, 1)

            # history_job_repr = self.causal_attention(history_jobs, sequence_mask_jobs)
            target_job = target_job.unsqueeze(1)  # [bsz, d_model] -> [bsz, 1, d_model]
            target_job_repr = self.mlp2(history_jobs.permute(0, 2, 1)).permute(0, 2, 1)

            concat_history = torch.cat([target_user_repr, target_job_repr], dim=-1)  # [bsz, 1, d_model * 2]
            concat_target = torch.cat([target_user, target_job], dim=-1)  # [bsz, 1, d_model * 2]
            concat = self.layer_norm(concat_history + concat_target)  # [bsz, 1, d_model * 2]
            logits = self.mlp1(concat).squeeze()  # [bsz]

        if labels is None:
            return {'logits': logits}
        else:
            loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}