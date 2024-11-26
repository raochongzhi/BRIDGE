import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gensim.models import word2vec, Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.modules"):
        '''
        param: sentences: the list of corpus
               sen_len: the max length of each sentence
               w2v_path: the path storing word emnbedding modules
        '''

        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word2vec modules ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        for i, word in enumerate(self.embedding.wv.vocab):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("")
        self.add_embedding("")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sentence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx[''])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        '''
        change words in sentences into idx in embedding_matrix
        '''
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx[''])
            sentence_idx = self.pad_sentence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        return torch.LongTensor(y)

# MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)        # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x

class JobLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.attn1 = SelfAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn2 = SelfAttentionEncoder(dim)

    def forward(self, x):
        # (N, S, L, D)
        x = x.permute(1, 0, 2, 3)   # (S, N, L, D)
        # print('x_1size', x.size())
        # x = self.attn1(x).unsqueeze(0)
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])   # (S, N, D)
        s = x.permute(1, 0, 2)      # (N, S, D)
        c = self.biLSTM(s)[0]       # (N, S, D)
        g = self.attn2(c)           # (N, D)
        return s, g

class CoAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=0)
        )

    def forward(self, x, s):
        # (N, L, D), (N, S2, D)
        s = s.permute(1, 0, 2)  # (S2, N, D)
        y = torch.cat([ self.attn( self.W(x.permute(1, 0, 2)) + self.U( _.expand(x.shape[1], _.shape[0], _.shape[1]) ) ).permute(2, 0, 1) for _ in s ]).permute(2, 0, 1)
        # (N, D) -> (L, N, D) -> (L, N, 1) -- softmax as L --> (L, N, 1) -> (1, L, N) -> (S2, L, N) -> (N, S2, L)
        sr = torch.cat([torch.mm(y[i], _).unsqueeze(0) for i, _ in enumerate(x)])   # (N, S2, D)
        sr = torch.mean(sr, dim=1)  # (N, D)
        return sr

class UserLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.co_attn = CoAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.self_attn = SelfAttentionEncoder(dim)

    def forward(self, x, s):
        # (N, S1, L, D), (N, S2, D)
        x = x.permute(1, 0, 2, 3)   # (S1, N, L, D)
        sr = torch.cat([self.co_attn(_, s).unsqueeze(0) for _ in x])   # (S1, N, D)
        u = sr.permute(1, 0, 2)     # (N, S1, D)
        c = self.biLSTM(u)[0]       # (N, S1, D)
        g = self.self_attn(c)       # (N, D)
        return g

class APJFNN(nn.Module):
    def __init__(self, embedding_job, embedding_user, fix_embedding=True):
        super(APJFNN, self).__init__()

        # 字典中有(行数)个词，词向量维度为(列数)
        self.user_embedding = nn.Embedding(embedding_user.size(0), embedding_user.size(1))
        self.user_embedding.weight = nn.Parameter(embedding_user)
        self.user_embedding.weight.requires_grad = False if fix_embedding else True

        self.job_embedding = nn.Embedding(embedding_job.size(0), embedding_job.size(1))
        self.job_embedding.weight = nn.Parameter(embedding_job)
        self.job_embedding.weight.requires_grad = False if fix_embedding else True

        dim = 200
        hd_size = 32
        self.user_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.job_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.job_layer = JobLayer(hd_size * 2, hd_size)
        self.user_layer = UserLayer(hd_size * 2, hd_size)

        self.mlp = MLP(
            input_size=hd_size *2 * 3,
            output_size=1,
            dropout=0.7
        )

    def forward(self, job, user):
        user = self.user_embedding(user.long())  # .long()
        job = self.job_embedding(job.long())
        # print('usersize', user.size())
        # print('jobsize:', job.size())
        user_vecs = self.user_biLSTM(user)[0].unsqueeze(1)
        job_vecs = self.job_biLSTM(job)[0].unsqueeze(1)
        # print('user_vecs_size', user_vecs.size())
        # print('job_vecs_size:', job_vecs.size())
        sj, gj = self.job_layer(job_vecs)
        gr = self.user_layer(user_vecs, sj)
        x = torch.cat([gj, gr, gj - gr], axis=1)
        x = self.mlp(x).squeeze(1)
        return x

class JobUserDataset(data.Dataset):
    def __init__(self, job, user, label):
        self.job = job
        self.user = user
        self.label = label

    def __getitem__(self, idx):
        if self.label is None:
            return self.job[idx], self.user[idx]
        return self.job[idx], self.user[idx], self.label[idx]

    def __len__(self):
        return len(self.job)

