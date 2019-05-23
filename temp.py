with open('test.txt', "r") as f:
    labels = []
    sentences = []
    dic = {}
    for line in f.readlines():
        a = line.strip().split("\t")
        labels.append(a[0])
        d=a[1].split(" ")
        sentences.append(d) 
    print(labels)
    print(type(f))

    
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
    print(dic2)

    sentences2 = []
    for sentence in sentences:
        sentence2=[]
        for word in sentence:
            if word in dic2.keys():
                sentence2.append(dic2[word])
            else:
                sentence2.append(dic2["unk"])
        sentences2.append(sentence2)
    print(sentences2)
    
    
 
