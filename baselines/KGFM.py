import torch
import torch.nn as nn

class KGFM(nn.Module):
    def __init__( self, n_users, n_entitys, n_relations, dim,
                  adj_entity, adj_relation, n_layers, agg_method,
                 mp_method, device, data_jobs, data_users, dropout_rate):
        super(KGFM, self).__init__()

        self.device = device
        self.n_layers = n_layers  # 消息传递的层数
        self.entity_embs = nn.Embedding(n_entitys, dim, max_norm=1)
        self.relation_embs = nn.Embedding(n_relations, dim, max_norm=1)

        # 定义线性层，将 768 维预训练的嵌入向量转换为 dim/2 维
        assert dim % 2 == 0, "Embedding dimension 'dim' 必须为偶数。"
        half_dim = dim // 2
        self.embedding_transform = nn.Linear(768, half_dim)

        # 将 data_users 和 data_jobs 转换为张量，并移动到设备
        # 假设 data_users 和 data_jobs 是 NumPy 数组或可转换为张量的数据结构
        data_users_tensor = torch.as_tensor(data_users, dtype=torch.float32)
        data_jobs_tensor = torch.as_tensor(data_jobs, dtype=torch.float32)

        # 通过线性层转换预训练的用户和职位嵌入
        transformed_data_users = self.embedding_transform(data_users_tensor)
        transformed_data_jobs = self.embedding_transform(data_jobs_tensor)

        # 保留原始随机初始化的另一半维度
        # 使用 .detach() 确保这些张量不会在计算图中追踪梯度
        random_users = self.entity_embs.weight.data[:n_users, half_dim:]
        random_jobs = self.entity_embs.weight.data[n_users:n_users + len(data_jobs), half_dim:]

        # 拼接用户和职位嵌入（dim/2 来自转换后的嵌入，dim/2 保持随机初始化）
        user_embeddings = torch.cat([transformed_data_users, random_users], dim=1)
        job_embeddings = torch.cat([transformed_data_jobs, random_jobs], dim=1)

        # 更新用户和职位的嵌入权重
        self.entity_embs.weight.data[:n_users, :] = user_embeddings
        self.entity_embs.weight.data[n_users:n_users + len(data_jobs), :] = job_embeddings

        self.adj_entity = adj_entity  # 节点的邻接列表
        self.adj_relation = adj_relation  # 关系的邻接列表

        self.agg_method = agg_method  # 聚合方法
        self.mp_method = mp_method  # 消息传递方法

        self.Wr1 = nn.Linear(dim, dim)
        self.Wr2 = nn.Linear(dim, dim)
        self.w = nn.Linear(768, dim)
        self.w_last1 = nn.Linear(dim * 4, dim)
        self.w_last2 = nn.Linear(dim, dim // 2)
        self.w_last3 = nn.Linear(dim // 2, 1)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        self.dropout = nn.Dropout(dropout_rate)

        if agg_method == 'concat':
            self.W_concat = nn.Linear(dim * 2, dim)
        else:
            self.W1 = nn.Linear(dim, dim)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(dim, dim)
        self.loss_fun = nn.BCEWithLogitsLoss()

    def get_neighbors(self, items):
        e_ids = [self.adj_entity[item] for item in items]
        r_ids = [self.adj_relation[item] for item in items]

        e_ids = torch.LongTensor(e_ids).to(self.device)
        r_ids = torch.LongTensor(r_ids).to(self.device)

        neighbor_entities_embs = self.entity_embs(e_ids)
        neighbor_relations_embs = self.relation_embs(r_ids)
        return neighbor_entities_embs, neighbor_relations_embs

    def FMMessagePassFromKGCN_item(self, u_embs, r_embs, t_embs):
        '''
        :param u_embs: 用户向量[ batch_size, dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, dim ]
        :param t_embs: 为实体向量[ batch_size, n_neibours, dim ]
        '''
        u_broadcast_embs = torch.cat([torch.unsqueeze(u_embs, 1) for _ in range(t_embs.shape[1])], dim=1)
        ur_embs = torch.sum(u_broadcast_embs * r_embs, dim=2)
        ur_embs = torch.softmax(ur_embs, dim=-1)
        ur_embs = torch.unsqueeze(ur_embs, 2)
        t_embs = ur_embs * t_embs
        square_of_sum = torch.sum(t_embs, dim=1) ** 2
        sum_of_square = torch.sum(t_embs ** 2, dim=1)
        output = square_of_sum - sum_of_square
        return output

    def FMMessagePassFromKGAT_item(self, h_embs, r_embs, t_embs):
        '''
        :param h_embs: 头实体向量[ batch_size, dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, dim ]
        :param t_embs: 为实体向量[ batch_size, n_neibours, dim ]
        '''
        h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs.shape[1])], dim=1)

        tr_embs = self.Wr1(t_embs)
        tr_embs = self.dropout(tr_embs)
        hr_embs = self.Wr1(h_broadcast_embs)
        hr_embs = self.dropout(hr_embs)
        hr_embs = torch.tanh(hr_embs + r_embs)

        hrt_embs = hr_embs * tr_embs

        hrt_embs = self.dropout(hrt_embs)

        square_of_sum = torch.sum(hrt_embs, dim=1) ** 2
        sum_of_square = torch.sum(hrt_embs ** 2, dim=1)

        output = square_of_sum - sum_of_square
        return output

    # KGAT由来的FM消息传递
    def FMMessagePassFromKGAT_user(self, h_embs, r_embs, t_embs):
        '''
        :param h_embs: 头实体向量[ batch_size, dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, dim ]
        :param t_embs: 为实体向量[ batch_size, n_neibours, dim ]
        '''
        h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs.shape[1])], dim=1)

        tr_embs = self.Wr2(t_embs)
        tr_embs = self.dropout(tr_embs)
        hr_embs = self.Wr2(h_broadcast_embs)
        hr_embs = self.dropout(hr_embs)
        hr_embs = torch.tanh(hr_embs + r_embs)

        hrt_embs = hr_embs * tr_embs

        hrt_embs = self.dropout(hrt_embs)

        square_of_sum = torch.sum(hrt_embs, dim=1) ** 2
        sum_of_square = torch.sum(hrt_embs ** 2, dim=1)

        output = square_of_sum - sum_of_square
        return output

    # 消息聚合
    def aggregate(self, h_embs, Nh_embs, agg_method='Bi-Interaction'):
        '''
        :param h_embs: 原始的头实体向量 [ batch_size, dim ]
        :param Nh_embs: 消息传递后头实体位置的向量 [ batch_size, dim ]
        :param agg_method: 聚合方式，总共有三种,分别是'Bi-Interaction','concat','sum'
        '''
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) \
                + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:  # sum
            return self.leakyRelu(self.W1(h_embs + Nh_embs))

    def forward(self, u, i, labels = None):
        u, i = u.to(self.device), i.to(self.device)
        t_embs, r_embs = self.get_neighbors(i)
        h_embs = self.entity_embs(i)
        user_embs = self.entity_embs(u)
        Nh_embs = self.FMMessagePassFromKGCN_item(user_embs, r_embs, t_embs)

        item_embs = self.aggregate(h_embs, Nh_embs, self.agg_method)
        user_embs = self.aggregate(user_embs, user_embs, self.agg_method)

        combined_features = torch.cat([user_embs, item_embs, user_embs + item_embs, user_embs * item_embs], dim=-1)

        # 最后一层计算输出
        logits = self.w_last1(combined_features)
        logits = self.w_last2(logits)
        logits = torch.sigmoid(self.w_last3(logits).squeeze(-1))

        if labels is None:
            return {'logits': logits}
        else:
            loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}