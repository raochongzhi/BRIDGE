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


class IPJF(torch.nn.Module):
    def __init__(self, word_embeddings1, word_embeddings2):
        super(IPJF, self).__init__()

        # embedding_matrix = [[0...0], [...], ...[]]
        self.Word_Embeds1 = torch.nn.Embedding.from_pretrained(word_embeddings1, padding_idx=0)
        self.Word_Embeds1.weight.requires_grad = False

        self.Word_Embeds2 = torch.nn.Embedding.from_pretrained(word_embeddings2, padding_idx=0)
        self.Word_Embeds2.weight.requires_grad = False

        self.Expect_ConvNet = torch.nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=5),
            # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3),

            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            # torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=5),
            # # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=50)
        )

        self.Job_ConvNet = torch.nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=5),
            # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2),

            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            # torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=3),
            # # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=50)
        )

        # match mlp
        self.Match_MLP = torch.nn.Sequential(
            torch.nn.Linear(2 * EMBED_DIM, EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(EMBED_DIM, 1),
            torch.nn.Sigmoid()
        )

    # [batch_size *2, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50)
    # term: padding same, word: padding 0
    # expects_sample, jobs_sample are in same format
    def forward(self, expects, jobs):
        # word level:
        # [batch_size, MAX_PROFILELEN, MAX_TERMLEN] (40, 15, 50) ->
        # [batch_size, MAX_PROFILELEN * MAX_TERMLEN](40 * 15, 50)
        shape = expects.shape
        expects_, jobs_ = expects.view([shape[0], -1]), jobs.view([shape[0], -1])

        # embeddings: [batch_size, MAX_PROFILELEN * MAX_TERMLEN, EMBED_DIM]

        jobs_wordembed = self.Word_Embeds1(jobs_).float()

        expects_wordembed = self.Word_Embeds2(expects_).float()

        # permute for conv1d
        # embeddings: [batch_size, EMBED_DIM, MAX_PROFILELEN * MAX_TERMLEN]
        expects_wordembed_ = expects_wordembed.permute(0, 2, 1)
        jobs_wordembed_ = jobs_wordembed.permute(0, 2, 1)

        # [batch_size, EMBED_DIM, x]
        expect_convs_out = self.Expect_ConvNet(expects_wordembed_)
        job_convs_out = self.Job_ConvNet(jobs_wordembed_)

        # [batch_size, EMBED_DIM, x] -> [batch_size, EMBED_DIM, 1]
        expect_len, job_len = expect_convs_out.shape[-1], job_convs_out.shape[-1]
        expect_final_out = torch.nn.AvgPool1d(kernel_size=expect_len)(expect_convs_out).squeeze(-1)
        job_final_out = torch.nn.MaxPool1d(kernel_size=job_len)(job_convs_out).squeeze(-1)

        return self.Match_MLP(torch.cat([expect_final_out, job_final_out], dim=-1)).squeeze(-1)

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