import numpy as np
import pandas as pd
import cv2
import os
import time



def readfile(path,label):
    img_dir = sorted(os.listdir(path))
    print(img_dir[0])
    x = np.zeros((len(img_dir),128,128,3),dtype = np.uint8) # 图片存在这个维度
    y = np.zeros(len(img_dir),dtype = np.uint8) # 标签
    for i , file in enumerate(img_dir):
        x[i] = cv2.resize(cv2.imread(os.path.join(path,file)),(128,128)) # 图像统一resize为128*128
        # print(i)
        # print(x.shape)
        # print(x[0].shape)
        # print(x[i,:,:].shape)
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return y

workspace_dir = "/Users/rivaille/Documents/bilibili/lihongyi_2020/3_cnn/food-11/"

print("Reading data")
train_x , train_y = readfile(os.path.join(workspace_dir,"training"),True)
print("Size of training data = {}".format(len(train_x)))
val_x , val_y = readfile(os.path.join(workspace_dir,"validation"),True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir,'testing'),False)
print("Size of Testing data = {}".format(len(test_x)))

# For training data argumentation
import torchvision.transforms as transforms
# from torchvision import datasets, transforms #(这里的datasets包含了mnist之类的数据集)
import torch

train_transform = transforms.Compose([
    transforms.ToPILImage(), #用于将Tensor变量的数据转换成PIL图片数据，主要是为了方便图片内容的显示
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor (C,H,W)，並把數值 normalize 到 [0,1] (data normalization), 即//225操作
])

# For testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# 继承Dataset，并且重写2个方法
# Dataset 需要 overload 兩個函數：__len__ 及 __getitem__
#
# __len__ 必須要回傳 dataset 的大小，而 __getitem__ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。
#
# 實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。
# Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
from torch.utils.data import DataLoader,Dataset,IterableDataset
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y) #64位整型
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None: #transform 不为空则做transform然后返回
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 128

# 实例对象
train_set = ImgDataset(train_x,train_y,train_transform) # 实例化train_set
print(train_set)
if isinstance(train_set, IterableDataset):
    print("train_set isinstance IterableDataset")
if isinstance(train_set, Dataset):
    print("train_set isinstance Dataset")
val_set = ImgDataset(val_x,val_y,test_transform)
# 第一个参数要么是torch.utils.data.Dataset类的对象，要么是继承自torch.utils.data.Dataset类的自定义类的对象
train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
val_loader = DataLoader(val_set,batch_size = batch_size,shuffle = False)

# exit()

# MODEL setting
import torch.nn as nn

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()  # 第一句话，调用父类的构造函数
        # 接下来定义myCNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64,128,128] # in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64,64,64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # flatten 本例子的话把 512*4*4 拉成 512*16
        return self.fc(out)

# Training
model = myCNN().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
num_epoch = 1
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # model.train() ：启用BatchNormalization和Dropout
    # model.eval() ：不启用BatchNormalization和Dropout
    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        print(train_pred.cpu().data.numpy().shape)
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))
# [030/030] 17.06 sec(s) Train Acc: 0.834482 Loss: 0.003854 | Val Acc: 0.515743 loss: 0.016422


###
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
#將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
