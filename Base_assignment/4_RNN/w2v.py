'''
读取数据
w2v.py
這個 block 是用來訓練 word to vector 的 word embedding
'''
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import Word2Vec

def load_training_data(path='training_label.txt'):
    if "training_label" in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            print(lines[0])
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        # print(x[0])
        # print(y[0])
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
        print(X[0])
    return X

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
    outputs[outputs<0.5] = 0 # 小於 0.5 為負面
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model
if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('training_label.txt')
    train_x_no_label = load_training_data('training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('testing_data.txt')

    # 下面这行使用了unlabal data
    model = train_word2vec(train_x + train_x_no_label + test_x)
    # model = train_word2vec(train_x + test_x)

    print("saving model ...")
    model.save(os.path.join('w2v_all.model'))
    # model.save(os.path.join('w2v_all.model'))