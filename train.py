from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from dataloader import MyDataLoader
from utils import word2id, id2embedding
from model import MLP


def main():
    parser = ArgumentParser(description='train a MLP model')
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('EMBED', type=str, help='path to embedding')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='gpu number')
    args = parser.parse_args()

    word_to_id = word2id(args.INPUT)
    embedding = id2embedding(args.EMBED, word_to_id)

    train_loader = MyDataLoader(args.INPUT, word_to_id,
                                batch_size=5000, shuffle=True, num_workers=1)
    # インスタンスを作成
    net = MLP(word_to_id, embedding)
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)

    gpu_id = args.gpu
    device = torch.device("cuda:{}".format(gpu_id) if gpu_id >= 0 else "cpu")
    net = net.to(device)

    epochs = 5
    log_interval = 10
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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               batch_idx * len(ids),
                                                                               len(train_loader.dataset),
                                                                               10 * batch_idx / len(train_loader),
                                                                               loss.item()))


if __name__ == '__main__':
    main()
