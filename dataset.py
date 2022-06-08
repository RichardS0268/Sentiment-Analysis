from torch.utils.data import Dataset
import numpy as np
import torch

class SA_Dataset(Dataset):
    def __init__(self, file_path, embedding, embedding_len, padding): # 50, 64
        super(SA_Dataset, self).__init__()
        self.embedding_len = embedding_len
        self.embedding = embedding
        self.padding = padding
        self.list = []
        self.statistic = np.zeros(2, dtype=np.int32)
        with open(file_path, "r") as f:   
            for line in f.readlines():
                items = line.split("\t")
                data = items[1].split()
                label = [int(items[0]), 1-int(items[0])] 
                self.list.append([data, label])

    def __getitem__(self, index):
        words = self.list[index][0]
        mat = []
        for x in words:
            if x in self.embedding:
                mat.append(self.embedding[x])
            else:
                mat.append([0]*self.embedding_len) # 如果不存在这个词就视为词向量是0，可能有更好的方法
        if (len(mat) > self.padding): # word长度64
            mat = mat[:self.padding]
        data = torch.cat([torch.FloatTensor(np.array(mat)),
                           torch.FloatTensor(np.zeros([self.padding-len(mat), self.embedding_len]))]) # 句子长度不足64字则补齐
        label = torch.tensor(np.array(self.list[index][1]))
        return data, label , len(mat)# data 64*50

    def __len__(self):
        return len(self.list)