import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 word2id
                 ) -> None:
        self.ids2, self.labels = self._load(file_path, word2id)

    def __len__(self) -> int:
        return len(self.ids2)

    # あるindexが与えられ、そのindexのデータを取り出す
    def __getitem__(self,
                    index: int
                    ):
        label  = self.labels[index]
        ids = self.ids2[index]
        mask= [1] * len(ids)
        return ids, mask, label

    def _load(self,
              file_path: str,
              word2id: dict
              ):
        with open(file_path, "r") as f:
            labels = []
            ids2 = []
            for line in f.readlines():
                label, sentence = line.strip().split("\t")
                labels.append(int(label == '1'))
                words = sentence.split(" ")
                ids = []
                for word in words:
                    if word in word2id.keys():
                        ids.append(word2id[word])
                    else:
                        ids.append(word2id["unk"])
                ids2.append(ids)
        return ids2, labels


class MyDataLoader(DataLoader):
    def __init__(self,
                 file_path,
                 word2id,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int
                 ) -> None:
        self.dataset = MyDataset(file_path, word2id)
        super(MyDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=my_collate_fn)

def my_collate_fn(batch
                  ):
    sources, masks, targets = [], [], []
    max_seq_len_in_batch = max(len(sample[0]) for sample in batch)
    for sample in batch:
        ids, mask, label = sample
        source_length = len(ids)
        source_padding = [0] * (max_seq_len_in_batch - source_length)
        source_mask_padding = [0] * (max_seq_len_in_batch - source_length)
        sources.append(ids + source_padding)
        masks.append(mask + source_mask_padding)
        targets.append(label)
    return torch.LongTensor(sources), torch.LongTensor(masks), torch.LongTensor(targets)