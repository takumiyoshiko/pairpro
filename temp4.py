import torch
import torch.nn as nn
import torch.nn.functional as F

with open('data/test.txt', "r") as f:
    labels = []
    sentences = []
    dic = {}
    for line in f.readlines():
        a = line.strip().split("\t")
        labels.append(a[0])
        d = a[1].split(" ")
        sentences.append(d)

    for sentence in sentences:
        for word in sentence:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    A = sorted(dic.items(), key=lambda x: x[1], reverse=True)

dic2 = {"pad":0,"unk":1 }
ID = 2
for pair in A:
    dic2[pair[0]] = ID
    ID += 1


from gensim.models import KeyedVectors
import numpy as np

w2v = KeyedVectors.load_word2vec_format("data/w2v.midasi.256.100K.bin", binary=True)
shape = (len(dic2), w2v.vector_size)
embedding = np.zeros(shape)
for word, id in dic2.items():
    if word in w2v.vocab:
        embedding[id] = w2v.word_vec(word)
    else:
        embedding[id] = w2v.word_vec("<UNK>")


class MLP(nn.Module):
    def __init__(self, n_in=256, n_mid_units=128, n_out=2):
        # お約束
        super(MLP, self).__init__()
        self.n_in = n_in
        # パラメータを持つ層を登録
        # Linear は W x + b の線形変換 (全結合) を行う
        # 第1引数は入力の次元数
        # 第2引数は出力の次元数
        self.l1 = nn.Linear(n_in, n_mid_units)
        self.l2 = nn.Linear(n_mid_units, n_out)
        self.tanh = nn.Tanh()
        self.embd = nn.Embedding(len(dic2),n_in)
        self._set_initial_embeddings(embedding)

    def forward(self, x, mask):
        # データを受け取った際のforward計算を書く
        # relu は Rectified Linear Unit: f(x)=max(0,x)
        e  = self.embd(x)
        x2 = e.sum(dim=1)
        mask2 = mask.sum(dim=1, keepdim=True, dtype=torch.float32)
        x3 = x2/mask2
        h1 = self.tanh(self.l1(x3))
        h2 = self.l2(h1)
        return h2

    def _set_initial_embeddings(self, weights):
        self.embd.weight = nn.Parameter(torch.Tensor(weights), requires_grad=False)

from temp3 import MyDataLoader


train_loader = MyDataLoader("data/test.txt",dic2,10,True,1)
# インスタンスを作成
net = MLP()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)

gpu_id = -1
device = torch.device("cuda:{}".format(gpu_id) if gpu_id >= 0 else "cpu")
net = net.to(device)

epochs = 5
log_interval = 100
for epoch in range(1, epochs + 1):
    net.train()  # おまじない (Dropout などを使う場合に効く)
    for batch_idx, (ids, mask, labels) in enumerate(train_loader):
        # data shape: (batchsize, 1, 28, 28)

        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()  # 最初に gradient をゼロで初期化; これを呼び出さないと過去の gradient が蓄積されていく
        output = net(ids,mask)
        output2 = F.softmax(output, dim=1)
        loss = F.binary_cross_entropy(output2[:, 1], labels.float())  # 損失を計算
        loss.backward()
        optimizer.step()  # パラメータを更新

        # 途中経過の表示
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(ids), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))