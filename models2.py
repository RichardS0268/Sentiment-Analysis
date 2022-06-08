import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


# -- MLP models
class MLP(nn.Module):
    def __init__(self, embedding, padding, opt):
        super(MLP, self).__init__()
        self.args = {
            'hidden1': 1024,
            'hidden2': 256,
            'dropoutRate': 0.3,
            'padding': padding,
            'classNum': 2
        }
        print(self.args)
        self.dropout = nn.Dropout(self.args['dropoutRate'])
        self.fc1 = nn.Linear(embedding * padding, self.args['hidden1']) # 输入维度 50 * 64
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.args['hidden1'], self.args['hidden2'])
        self.fc3 = nn.Linear(self.args['hidden2'], self.args['classNum'])
     
    def forward(self, x, _):
        x = x.view(x.shape[0], -1) # 256 * 3200
        x = self.fc1(x) # 256 * 1024
        x = self.dropout(x) 
        x = self.sigmoid(x) 
        x = self.fc2(x) # 256 * 256
        x = self.sigmoid(x) 
        x = self.fc3(x)
        return x


class MMPMLP(nn.Module):
    def __init__(self, embedding, padding, opt):
        super(MMPMLP, self).__init__()
        self.args = {
            'hidden1': 1024,
            'hidden2': 256,
            'dropoutRate': 0.1,
            'classNum': 2
        }
        print(self.args)
        self.dropout = nn.Dropout(self.args['dropoutRate'])
        self.fc1 = nn.Linear(embedding*2, self.args['hidden1'])
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.args['hidden1'], self.args['hidden2'])
        self.fc3 = nn.Linear(self.args['hidden2'], self.args['classNum'])

    
    def forward(self, x, _): # (bs, padding, embedding)
        x = torch.cat([torch.max(x, dim=1)[0], torch.min(x, dim=1)[0]], dim=1) # 256 * 100
        x = self.fc1(x) # 256 * 1024
        x = self.dropout(x) 
        x = self.sigmoid(x)
        x = self.fc2(x) # 256 * 256
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


# -- CNN models
class CNN(nn.Module):
    def __init__(self, embedding, _, opt):
        super(CNN, self).__init__()
        self.args = {
            'ci': 1,
            'co': 256,
            'kernalSize': [3, 4, 5],
            'dropoutRate': 0.5,
            'classNum': 2
        }
        print(self.args)
        self.conv = nn.ModuleList([nn.Conv2d(self.args['ci'], self.args['co'], [ks, embedding]) for ks in self.args['kernalSize']])
        self.dropout = nn.Dropout(self.args['dropoutRate'])
        self.fc = nn.Linear(len(self.args['kernalSize']) * self.args['co'], self.args['classNum'])
        self.activate = nn.ReLU()


    def forward(self, x, _):
        x = x.unsqueeze(1) # (bs, 1, padding, embedding)
        x = [conv(x) for conv in self.conv] # (bs, co, padding, 1) * len(KS)
        x = [self.activate(i.squeeze(3)) for i in x] # (bs, co, padding) * len(KS)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # (bs, co) * len(KS)
        x = [self.dropout(y) for y in x]
        x = torch.cat(x, 1) # (bs, co * KS)
        # x = self.dropout(x) # (bs, co * KS)
        x = self.fc(x) # (bs, classNum)
        return x


class BasicBlock(nn.Module):
    def __init__(self, ci, co, kernalSize, stride):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(co, co, kernalSize, padding = stride)
        self.relu = nn.PReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, embedding, _, opt):
        super(DeepCNN, self).__init__()
        self.args = {
            'dep': 5,
            'ci': 1,
            'co': 128,
            'kernalSize': 3,
            'stride': 1,
            'classNum': 2,
            'dropoutRate': 0.3
        }
        print(self.args)
        self.conv0 = nn.Conv2d(self.args['ci'], self.args['co'], [self.args['kernalSize'], embedding], padding = [self.args['stride'], 0])
        self.pool0 = nn.MaxPool1d(2)
        self.relu0 = nn.ReLU()
        self.conv1 = BasicBlock(self.args['co'], self.args['co'], self.args['kernalSize'], self.args['stride'])
        self.conv2 = BasicBlock(self.args['co'], self.args['co'], self.args['kernalSize'], self.args['stride'])
        self.conv3 = BasicBlock(self.args['co'], self.args['co'], self.args['kernalSize'], self.args['stride'])
        self.conv4 = BasicBlock(self.args['co'], self.args['co'], self.args['kernalSize'], self.args['stride'])
        self.conv5 = BasicBlock(self.args['co'], self.args['co'], self.args['kernalSize'], self.args['stride'])
        self.dropout = nn.Dropout(self.args['dropoutRate'])
        self.fc = nn.Linear(self.args['co'], self.args['classNum'])

    def forward(self, x, _): 
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = self.conv0(x)
        x = x.squeeze(3)
        x = self.pool0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        if (self.args['dep'] > 1):
            x = self.conv2(x)
        if (self.args['dep'] > 2):
            x = self.conv3(x)
        if (self.args['dep'] > 3):
            x = self.conv4(x)
        if (self.args['dep'] > 4):
            x = self.conv5(x)
        x = x.view([x.shape[0], -1])
        x = self.fc(x)
        return x


# -- RNN models
class RNN(nn.Module):
    def __init__(self, embedding, _, opt):
        super(RNN, self).__init__()
        self.args = {
            'hidden': 128,
            'num_layers': 3,
            'dropoutRate': 0.7,
            'bidirectional': True,
            'classNum': 2
        }
        print(self.args)
        self.lstm = nn.LSTM(input_size = embedding,
                            hidden_size = self.args['hidden'], 
                            num_layers = self.args['num_layers'], 
                            bidirectional = True,
                            dropout = self.args['dropoutRate'])
        self.fc = nn.Linear(self.args['hidden']*self.args['num_layers']*2, self.args['classNum'])

    def forward(self, x, x_len): # (bs, padding, embedding) (bs)
        _, idx_sort = torch.sort(x_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x = x.index_select(0, Variable(idx_sort))
        x_padded = nn.utils.rnn.pack_padded_sequence(x, lengths=list(x_len[idx_sort]), batch_first=True)
        _, (ht, _) = self.lstm(x_padded)
        ht = ht.permute([1, 0, 2])
        ht = ht.index_select(0, Variable(idx_unsort))
        ht = ht.view([ht.shape[0], -1])
        output = self.fc(ht)
        return output



class MPRNN(nn.Module):
    def __init__(self, embedding, _, opt):
        super(MPRNN, self).__init__()
        self.args = {
            'hidden': 128,
            'num_layers': 3,
            'dropoutRate': 0.3,
            'bidirectional': True,
            'classNum': 2
        }
        print(self.args)
        self.lstm = nn.LSTM(input_size = embedding,
                            hidden_size = self.args['hidden'], 
                            num_layers = self.args['num_layers'], 
                            bidirectional = True,
                            dropout = self.args['dropoutRate'])
        self.fc = nn.Linear(self.args['hidden'] * 2, self.args['classNum'])

    def forward(self, x, x_len): # (bs, padding, embedding) (bs)
        _, idx_sort = torch.sort(x_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x = x.index_select(0, Variable(idx_sort))
        x_padded = nn.utils.rnn.pack_padded_sequence(x, lengths=list(x_len[idx_sort]), batch_first=True)
        states, (ht, _) = self.lstm(x_padded)
        states = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)[0]
        states = states.index_select(0, Variable(idx_unsort))
        states = torch.max(states, dim = 1)[0]
        
        output = self.fc(states)
        return output