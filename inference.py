# utils
from __future__ import print_function
import argparse
import os
import re
import time
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
# torch module
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
# self-defined module
import models2 as Model_set
from dataset import SA_Dataset

# -- Training settings
parser = argparse.ArgumentParser(description='PyTorch Sentiment Analysis') 
parser.add_argument('--model', type=str, default='MLP', help='model name') # TODO: model's name
parser.add_argument('--bs', type=int, default=256, help='training batch size') # TODO: rnns 64, cnns & mlps 256 
parser.add_argument('--padding', type=int, default=64, help='length of cases after padding') 
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--test', type=str, default='./Dataset/test.txt', help='path to testing dataset')
parser.add_argument('--embedding', type=str, default='./Dataset/wiki_word2vec_50.bin', help='path to word embedding')
parser.add_argument('--embedding_len', type=int, default=50, help='embedding length') 
parser.add_argument('--model_output', type=str, default='model', help='model output dir')
parser.add_argument('--device', type=int, default=0, help='device rank') # TODO: parallel training on different device

options = parser.parse_args()
modelName = options.model+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

device = torch.device('cuda')
torch.cuda.set_device(options.device)


# -- Data preparation
print('[!] Load dataset ... ', end='', flush=True)
wv = KeyedVectors.load_word2vec_format(options.embedding, binary=True)

test_set = SA_Dataset(options.test, wv, options.embedding_len, options.padding)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.bs, shuffle=False, drop_last=False)
print('done !', flush=True)


def report(model, data_loader, model_name):
    dir = os.listdir(options.model_output)
    model_path = options.model_output + '/' + list(filter(lambda x: re.match(model_name + '*', x)!=None, dir))[0]
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
    print(f"[!] Test Acc: {round(acc.item(), 4)}" )
    print(f"[!] F-score: {round(F1.item(), 4)}" )


if __name__ == '__main__':
    pretrained_models = []
    if options.model == 'all':
        pretrained_models = ['MLP', 'MMPMLP', 'CNN', 'DeepCNN', 'RNN', 'MPRNN']
    else:
        pretrained_models.append(options.model)

    for model_name in pretrained_models:
        print(f"- {model_name}")
        model = getattr(Model_set, model_name)(options.embedding_len, options.padding, options)
        model = model.cuda()
        report(model, test_data_loader, model_name)
