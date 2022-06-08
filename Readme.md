# 🥰文本情感分类

## **1. 网络结构**

实现CNN，RNN，MLP三类模型，每一类模型尝试不同的网络结构。

### 1.1 MLP

#### BasicMLP 

将词向量拉成一维，通过隐藏层、激活函数、最后进入分类器。

```python
# 前向传播过程
def forward(self, x, _):
    x = x.view(x.shape[0], -1)
    x = self.fc1(x) 
    x = self.dropout(x) 
    x = self.sigmoid(x) 
    x = self.fc2(x) 
    x = self.sigmoid(x) 
    x = self.fc3(x)
    return x
```
<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled.png" alt="图1：BasicMLP 网络结构" width = "600px" />
图1：BasicMLP 网络结构


#### MMPMLP (min-max pooling MLP)

对词向量的每一维，分别取输入词向量的最大值，最小值。将合成向量通过全连接层、激活函数、分类器。

```python
# 前向传播过程
def forward(self, x, _): 
    x = torch.cat([torch.max(x, dim=1)[0], torch.min(x, dim=1)[0]], dim=1) 
    x = self.fc1(x)
    x = self.dropout(x) 
    x = self.sigmoid(x)
    x = self.fc2(x) 
    x = self.sigmoid(x)
    x = self.fc3(x)
    return x
```

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%201.png" alt="图2：MMPMLP网络结构" width = "600px"  />
图2：MMPMLP网络结构

### 1.2 CNNs

#### SimpleCNN

采用 Yoon Kim模型。

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%202.png" alt="图3：SimpleCNN 网络结构" style="zoom: 67%;" />
图3：SimpleCNN 网络结构

```python
# 前向传播过程 
def forward(self, x, _):
    x = x.unsqueeze(1)
    x = [conv(x) for conv in self.conv]
    x = [self.activate(i.squeeze(3)) for i in x]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = [self.dropout(y) for y in x]
    x = torch.cat(x, 1)
    x = self.fc(x)
    return x
```

#### DeepCNN

与传统的CNN类似，在网络中多次堆叠卷积层、激活函数、缩小一倍的池化层。最后用一个全连接层进行分类。

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%203.png" alt="图4：DeepCNN 网络结构" style="zoom: 67%;" />
图4：DeepCNN 网络结构

```python
# 前向传播过程
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
```

### 1.3 RNNs

#### BasicRNN

将词向量一次输入循环神经网络，把循环神经网络最后一个时刻的输出输入分类器。

```python
# 前向传播过程
def forward(self, x, x_len): 
	...
    x_padded = nn.utils.rnn.pack_padded_sequence(x, lengths=list(x_len[idx_sort]), batch_first=True)
    _, (ht, _) = self.lstm(x_padded)
    ht = ht.permute([1, 0, 2])
    ht = ht.index_select(0, Variable(idx_unsort))
    ht = ht.view([ht.shape[0], -1])
    output = self.fc(ht)
    return output
```

#### MPRNN

将词向量依次输入循环神经网络，取每个时刻输出的最大值进入分类器。

```python
# 前向传播过程
def forward(self, x, x_len): 
        ...
    x_padded = nn.utils.rnn.pack_padded_sequence(x, lengths=list(x_len[idx_sort]), batch_first=True)
    states, (ht, _) = self.lstm(x_padded)
    states = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)[0]
    states = states.index_select(0, Variable(idx_unsort))
    states = torch.max(states, dim = 1)[0]
    output = self.fc(states)
    return output
```

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%204.png" alt="图4：DeepCNN 网络结构" style="zoom: 67%;" />
图5：BasicRNN 及 MPRNN 网络结构 

## 2. 实现细节

### 2.1 输入数据

pytorch 的 `dataloader` 要求数据的位数严格一致，因此需要对较长的句子要进行截取，对较短的句子则要用0补全。如图为训练集，验证集和测试集中句子长度（分词后词语的数量）的分布。为了尽可能保留文章的全部信息，同时又不至于太大的计算开销，选取句子长度为64，由图可知这样可以覆盖超过97%的数据。

对于句中一些没有出现在 `word2vec` 中的词语，将这些位置填上0。

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%205.png" alt="图6：训练集句长分布" width = "600px"  />
图6：训练集句长分布


<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%206.png" alt="图7：验证集句长分布"  width = "600px"/>
图7：验证集句长分布


<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%207.png" alt="图8：测试集句长分布" width = "600px" />
图8：测试集句长分布


### 2.2 RNN

在RNN中，使用 `pack_padded_sequence` 填补0。该接口需要batch中各个数据的句长，为了实现整个pipline的通用性，在各个模型的前向过程中均加入参数 `input_len`

### 2.3 学习率递减

随着训练过程的进行，减小学习率有利于模型的收敛。

通过`torch.optim.lr_scheduler.LambdaLR` 实现学习率递减，其中 `--dr` 为递减速率，实验中RNNs选择0.5，其他模型选择0.95。

```python
lambda1 = lambda epoch: options.dr ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
```

### 2.4 早停机制

通过参数 `--early_stop` 及 `--max_epoch` 来决定是否进行训练早停（尚未达到设定的epoch数就停止训练）以及规定所容忍的模型性能连续没有提升的epoch数（即模型在validation set上准确率不提升）。

通过早停机制及`--max_epoch` 的设定，可缩短训练时间，节约训练资源，同时使模型精度的变化在容忍范围内。

### 2.5 模型保存与测试

对于每个epoch，当模型在validation set上的准确率达到新高时就保存该模型，同时删除之前的模型。当训练早停或到达指定 epoch 后，通过`report(model, test_data_loader)`函数加载训练过程中在验证集中表现最佳的模型，测试其在测试集上的准确率并计算 F-score。

## 3. 训练与加载

### 3.1 预训练模型

```bash
# MLPs
python inference.py --model MLP
python inference.py --model MMPMLP
#CNNs
python inference.py --model CNN
python inference.py --model DeepCNN
#RNNs
python inference.py --model RNN
python inference.py --model MPRNN
# all models
python inference.py --model all
```

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%208.png" alt="图6：训练集句长分布" style="zoom: 67%;" />

### 3.2 训练模型

```bash
# MLPs
python main.py --model MLP --bs 256 --lr 0.0001 --dr 0.95 --early_stop 1 --max_epoch 5
python main.py --model MMPMLP --bs 256 --lr 0.0001 --dr 0.95 --early_stop 1 --max_epoch 5
# CNNs
python main.py --model CNN --bs 256 --lr 0.001 --dr 0.95 --early_stop 1 --max_epoch 5
python main.py --model DeepCNN --bs 256 --lr 0.001 --dr 0.95 --early_stop 1 --max_epoch 5
# RNNs
python main.py --model RNN --bs 64 --lr 0.0001 --dr 0.5 --early_stop 1 --max_epoch 5
python main.py --model MPRNN --bs 64 --lr 0.0001 --dr 0.5 --early_stop 1 --max_epoch 5
```

<img src="https://github.com/RichardS0268/Introduction-to-AI/blob/main/DeepLearning/Sentiment%20Analysis/docs/Untitled%209.png" alt="图6：训练集句长分布" style="zoom: 67%;" />

**Training Log 示例**

```
---------------------------------------------
[!] Training epoch 5 ...
 -  Current learning rate is 6.25e-06
 -  Epoch[5] (312/312): Loss: 0.4795
[!] Epoch 5 complete.
[!] Epoch 5: train_acc 15077/19968           
[!] Epoch 5: valid_acc 4221/5629           
[!] Epoch 5: best_acc 0.7527
---------------------------------------------
[!] Training epoch 6 ...
 -  Current learning rate is 3.13e-06
 -  Epoch[6] (312/312): Loss: 0.4850
[!] Epoch 6 complete.
[!] Epoch 6: train_acc 15107/19968           
[!] Epoch 6: valid_acc 4249/5629           
[!] Saving checkpoint into model/MPRNN2022-05-19_02-10-36_acc{0.7548}_epoch_6_2022-05-19_02-12-54.pth ... done !
[!] Epoch 6: best_acc 0.7548
---------------------------------------------
[!] Training epoch 7 ...
 -  Current learning rate is 1.56e-06
 -  Epoch[7] (312/312): Loss: 0.5728
[!] Epoch 7 complete.
[!] Epoch 7: train_acc 15112/19968           
[!] Epoch 7: valid_acc 4251/5629           
[!] Saving checkpoint into model/MPRNN2022-05-19_02-10-36_acc{0.7552}_epoch_7_2022-05-19_02-13-17.pth ... done !
[!] Epoch 7: best_acc 0.7552
---------------------------------------------
[!] Training epoch 8 ...
 -  Current learning rate is 7.8e-07
 -  Epoch[8] (312/312): Loss: 0.5301
[!] Epoch 8 complete.
[!] Epoch 8: train_acc 15107/19968           
[!] Epoch 8: valid_acc 4251/5629           
[!] Epoch 8: best_acc 0.7552
---------------------------------------------
```
