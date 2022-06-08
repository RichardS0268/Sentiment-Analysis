# 🥰 Sentiment-Analysis
### 1. Pre-trained Models

```
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

Model Performace

| Models     | acc    | dropout rate | F-score |
| ---------- | ------ | ------------ | ------- |
| Basic-MLP  | 0.7453 | 0.3          | 0.7526  |
| MP-MLP     | 0.7100 | 0.1          | 0.7084  |
| Simple-CNN | 0.8266 | 0.5          | 0.8307  |
| Deep-CNN   | 0.7507 | 0.3          | 0.7527  |
| Basic-RNN  | 0.7642 | 0.7          | 0.7692  |
| MP-RNN     | 0.7669 | 0.3          | 0.7737  |

### 2. Train Models

```sh
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

```sh
# training log 
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

