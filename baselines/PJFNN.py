import torch
from torch.utils import data
import torch.nn as nn
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

class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size[0]),
            nn.BatchNorm2d(channels), # 其中的参数是通道数
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim)) # （1，dim）是指输出大小
        )
        if method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x).squeeze(2)
        x = self.pool(x).squeeze(1)
        return x

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

class PJFNN(nn.Module):
    def __init__(self, embedding1,embedding2, channels=1, dropout=0.5, fix_embedding=True):
        super(PJFNN, self).__init__()
        self.dim1 = embedding1.size(1)
#         self.user_dim = input_dim
        self.dim2 = embedding2.size(1)
        self.channels = channels
        # job  字典中有(行数)个词，词向量维度为(列数)
        self.embedding1 = nn.Embedding(embedding1.size(0), embedding1.size(1))
        self.embedding1.weight = nn.Parameter(embedding1)
        self.embedding1.weight.requires_grad = False if fix_embedding else True
        # user
        self.embedding2 = nn.Embedding(embedding2.size(0), embedding2.size(1))
        self.embedding2.weight = nn.Parameter(embedding2)
        self.embedding2.weight.requires_grad = False if fix_embedding else True

        self.linear_transform = nn.Linear(200, 64)

        self.geek_layer = TextCNN(
            channels=self.channels,
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=200,
            method='max'
        )

        self.job_layer = TextCNN(
            channels=self.channels,
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=200,
            method='mean'
        )

        self.mlp = MLP(
            input_size=128,
            output_size=1,
            dropout=dropout
        )


    def forward(self, job, user):
        job = self.embedding1(job.long()) #.long()
        job = job.unsqueeze(1)
        job = self.job_layer(job)

        user = self.embedding2(user.long()) #.long()
        user = user.unsqueeze(1)
        user = self.geek_layer(user)

        user = self.linear_transform(user)
        job = self.linear_transform(job)
        # tensor进行拼接
        x = torch.cat((user,job),dim=1)
        # mlp层
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