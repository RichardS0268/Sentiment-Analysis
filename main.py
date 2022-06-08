# utils
from __future__ import print_function
import argparse
import json
import os
import re
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
# torch module
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# self-defined module
import models2 as Model_set
from dataset import SA_Dataset

# -- Training settings
parser = argparse.ArgumentParser(description='PyTorch Sentiment Analysis') 
parser.add_argument('--model', type=str, default='CNN', help='model name') # TODO: model's name
parser.add_argument('--bs', type=int, default=256, help='training batch size') # TODO: rnns 64, cnns & mlps 256 
parser.add_argument('--padding', type=int, default=64, help='length of cases after padding') 
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01') # TODO: rnns & MMPMLP 0.0001, cnns & MLP 0.001
parser.add_argument('--dr', type=float, default=0.95, help='lr discount rate') # TODO: rnns 0.5, cnns & mlps 0.95
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=233, help='random seed to use. Default=223')
parser.add_argument('--test', type=str, default='./Dataset/test.txt', help='path to testing dataset')
parser.add_argument('--train', type=str, default='./Dataset/train.txt', help='path to training dataset')
parser.add_argument('--validation', type=str, default='./Dataset/validation.txt', help='path to validation dataset')
parser.add_argument('--embedding', type=str, default='./Dataset/wiki_word2vec_50.bin', help='path to word embedding')
parser.add_argument('--embedding_len', type=int, default=50, help='embedding length') 
parser.add_argument('--result', type=str, default='results', help='result dir')
parser.add_argument('--model_output', type=str, default='model', help='model output dir')
parser.add_argument('--device', type=int, default=0, help='device rank') # TODO: parallel training on different device
parser.add_argument('--metadata', type=str, default='metadata', help='path to model parameters')
parser.add_argument('--droprate', type=float, default=0, help='path to model parameters')
parser.add_argument('--early_stop', type=int, default=0, help='early stop(1) or not(0)')
parser.add_argument('--max_epoch', type=int, default=5, help='max epoch num having no improvement')
parser.add_argument('--early_stop_count', type=int, default=0, help='for early stop')
parser.add_argument('--load_model', type=int, default=0, help='load pretrained model(1) or not(0)')
parser.add_argument('--pth', type=str, default=" ", help='pretained model path')
options = parser.parse_args()
print(options)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

setup_seed(options.seed)
modelName = options.model+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

device = torch.device('cuda')
torch.cuda.set_device(options.device)

# -- Data preparation
print('[!] Load dataset ... ', end='', flush=True)

wv = KeyedVectors.load_word2vec_format(options.embedding, binary=True)

train_set = SA_Dataset(options.train, wv, options.embedding_len, options.padding)
valid_set = SA_Dataset(options.validation, wv, options.embedding_len, options.padding)
test_set = SA_Dataset(options.test, wv, options.embedding_len, options.padding)
train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=True, drop_last=True)
valid_data_loader = DataLoader(dataset=valid_set, num_workers=options.threads, batch_size=options.bs, shuffle=False, drop_last=False)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.bs, shuffle=False, drop_last=False)
print('done !', flush=True)

# -- Build Model
print('[!] Building model ... ', end='', flush=True)
model = getattr(Model_set, options.model)(options.embedding_len, options.padding, options)
model = model.cuda()
print('done !', flush=True)


# -- training process
optimizer = optim.Adam(model.parameters(), lr=options.lr)

lambda1 = lambda epoch: options.dr ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
criterion = nn.CrossEntropyLoss().to(device)

acc_max = 0
test_epochs = [0]
train_accs = [0]
test_accs = [0]
losses = []


def train(epoch):
    model.train()
    print('[!] Training epoch ' + str(epoch) + ' ...')
    print(' -  Current learning rate is ' + str(round(optimizer.param_groups[0]['lr'], 8)), flush=True)
    loss_sum = 0
    iter_num = 0
    for iteration, batch in enumerate(train_data_loader):
        iter_num = max(iter_num, iteration)
        input, target, input_len = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        input, target, input_len = Variable(input), Variable(target), Variable(input_len)
        optimizer.zero_grad()
        output = F.softmax(model(input, input_len), dim = 1)
        target = target.to(torch.float32) # tensor 256
        loss = criterion(target, output)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        print(' -  Epoch[{}] ({}/{}): Loss: {:.4f}\r'.format(epoch, iteration+1, len(train_data_loader), loss.item()), end='', flush=True)
    losses.append(loss_sum / (iter_num + 1))
    if epoch == 1:
        losses.append(loss_sum / (iter_num + 1))
    print('\n[!] Epoch {} complete.'.format(epoch))
    scheduler.step()


# -- test (report loss of training and validation)
def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    result = []
    std = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            input, target, input_len = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            input, target, input_len = Variable(input), Variable(target), Variable(input_len)
            output = model(input, input_len)
            output = torch.max(output, 1)[1]
            target = torch.max(target, 1)[1]
            correct += torch.sum(output==target).item()
            total += input.shape[0]
            result.append(output)
            std.append(target)
        result = torch.cat(result)
        std = torch.cat(std)
    return result, std, correct, total


# -- report (final acc on test_data_set)
def report(model, data_loader):
    dir = os.listdir(options.model_output)
    model_path = options.model_output + '/' + list(filter(lambda x: re.match(options.model + '*', x)!=None, dir))[0]
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    TP = TN = FN = FP = 0
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            input, target, input_len = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            input, target, input_len = Variable(input), Variable(target), Variable(input_len)
            output = model(input, input_len)
            output = torch.max(output, 1)[1]
            target = torch.max(target, 1)[1]
            # TP    predict 和 label 同时为1
            TP += ((output == 1) & (target == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((output == 0) & (target == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((output == 0) & (target == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((output == 1) & (target == 0)).cpu().sum()

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
    # print(f"[!] Test Acc: {round(correct*1.0/total, 4)}" )
    print(f"[!] Test Acc: {round(acc.item(), 4)}" )
    print(f"[!] F-score: {round(F1.item(), 4)}" )


# -- save model and results
def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def checkpoint(epoch, model, correct, total):
    out_path = options.model_output + '/' + modelName + '_acc{%.4f}'%(correct*1.0/total) + '_epoch_{}'.format(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth'
    try:
        os.system(f"rm {options.model_output + '/' + options.model}**.pth")
    except:
        print("No Model yet")
    print('[!] Saving checkpoint into ' + out_path + ' ... ', flush=True, end='')
    save_model(model, out_path)
    print('done !', flush=True)


def save_metadata():
    if not options.model in os.listdir(options.metadata):
        os.system(f"mkdir {options.metadata}/{options.model}")
    with open(options.metadata + '/' + options.model + '/' + modelName + '.json', 'w') as f:
        json.dump(model.args, f)


def save_mid():
    with open(options.metadata + '/' + options.model + '/' + modelName + '.csv', 'w') as f:
        f.write(','.join([str(x) for x in test_epochs]) + '\n')
        f.write(','.join([str(x) for x in train_accs]) + '\n')
        f.write(','.join([str(x) for x in test_accs]) + '\n')
        f.write(','.join([str(x) for x in losses]) + '\n')
    plt.figure(figsize=(10, 5), dpi = 150)
    plt.plot(test_epochs, train_accs, label='train')
    plt.plot(test_epochs, test_accs, label='test')
    plt.vlines(last_epoch, test_accs[last_epoch]-0.001, train_accs[last_epoch]+0.001, colors='r', linestyles='--') # 画布横坐标的起始值为31
    plt.xlabel("Train epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(options.metadata + '/' + options.model + '/' + modelName + '.png')
    plt.clf()
    plt.plot(test_epochs, losses)
    plt.xlabel("Train epoch")
    plt.ylabel("Train Loss")
    plt.savefig(options.metadata + '/' + options.model + '/' + modelName + '_loss.png')


def valid_and_save(epoch):
    global acc_max
    global last_epoch
    train_output, train_target, train_correct, train_total = test(model, train_data_loader)
    print('[!] Epoch {}: train_acc {}/{}           '.format(epoch, train_correct, train_total))
    valid_output, valid_target, valid_correct, valid_total = test(model, valid_data_loader)
    print('[!] Epoch {}: valid_acc {}/{}           '.format(epoch, valid_correct, valid_total))
    train_acc = train_correct*1.0/train_total
    valid_acc = valid_correct*1.0/valid_total
    if (valid_acc > acc_max):
        acc_max = valid_acc 
        checkpoint(epoch, model, valid_correct, valid_total)
        options.early_stop_count = 0
        last_epoch = epoch
    else:
        options.early_stop_count += 1
    print(f'[!] Epoch {epoch}: best_acc {round(acc_max, 4)}')
    #output_result(epoch, valid_output, valid_target)
    print("---------------------------------------------")
    test_epochs.append(epoch)
    train_accs.append(train_acc)
    test_accs.append(valid_acc)


if __name__ == '__main__':
    save_metadata()
    for epoch in range(1, options.epochs):
        train(epoch)
        valid_and_save(epoch) 
        if (options.early_stop) and (options.early_stop_count == options.max_epoch):
            print(f"[!] Having not improved for {options.max_epoch} epoches, Early stop!")
            break
    report(model, test_data_loader)
    save_mid()