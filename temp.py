with open('data/test.txt', "r") as f:
    labels = []
    sentences = []
    dic = {}
    for line in f.readlines():
        a = line.strip().split("\t")
        labels.append(a[0])
        d=a[1].split(" ")
        sentences.append(d)
    
    for sentence in sentences:
        for word in sentence:
            if word in dic.keys():
                dic[word] += 1 
            else:
                dic[word] = 1
    A = sorted(dic.items(), key=lambda x: x[1],reverse=True)

    dic2 = {"pad":0,"unk":1 }
    ID = 2
    for pair in A:
        dic2[pair[0]] = ID
        ID += 1

    sentences2 = []
    for sentence in sentences:
        sentence2=[]
        for word in sentence:
            if word in dic2.keys():
                sentence2.append(dic2[word])
            else:
                sentence2.append(dic2["unk"])
        sentences2.append(sentence2)
    
from gensim.models import KeyedVectors
import numpy as np


w2v = KeyedVectors.load_word2vec_format("data/w2v.midasi.256.100K.bin", binary=True)
shape=(len(dic2), w2v.vector_size)
embedding = np.zeros(shape)
for word, id in dic2.items():
    if word in w2v.vocab:
        embedding[id] = w2v.word_vec(word)
    else:
        embedding[id] = w2v.word_vec("<UNK>")

