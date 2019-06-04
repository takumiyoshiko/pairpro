from typing import Dict

from gensim.models import KeyedVectors
import numpy as np
import torch


def word2id(file_path: str
            ) -> Dict[str, int]:
    with open(file_path, "r") as f:
        labels = []
        sentences = []
        dic = {}
        for line in f.readlines():
            label, sentence = line.strip().split("\t")
            labels.append(label)
            words = sentence.split(" ")
            sentences.append(words)

        for sentence in sentences:
            for word in sentence:
                if word in dic.keys():
                    dic[word] += 1
                else:
                    dic[word] = 1
        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)

    word_to_id = {"pad": 0, "unk": 1}
    num = 2
    for pair in dic:
        word_to_id[pair[0]] = num
        num += 1
    return word_to_id


def id2embedding(file_path: str,
                 word_to_id: Dict[str, int]
                 ) -> torch.Tensor:
    w2v = KeyedVectors.load_word2vec_format(file_path, binary=True)
    shape=(len(word_to_id), w2v.vector_size)
    embedding = np.zeros(shape, dtype='f')
    for word, id in word_to_id.items():
        if word in w2v.vocab:
            embedding[id] = w2v.word_vec(word)
        else:
            embedding[id] = w2v.word_vec("<UNK>")
    return torch.tensor(embedding)
