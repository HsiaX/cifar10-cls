import sys
import os
import time
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from runner import SaveWeight, Logger
from model import alexnet, CifarConfig
from datasets import Cifar10Dataset
from torch.utils.data import DataLoader


# 加载配置信息，修改默认参数
config = CifarConfig()
config.lr = 0.01
config.batch_size = 64
config.epoch = 5

# 加载数据集
TrainDataset = Cifar10Dataset(img_path = "C:\\code\\cifar10-cls\\datasets\\",
                                  json_path = "C:\\code\\cifar10-cls\\datasets\\",
                                  train=True, transform=torchvision.transforms.ToTensor())
TestDataset = Cifar10Dataset(img_path="C:\\code\\cifar10-cls\\datasets\\",
                                  json_path="C:\\code\\cifar10-cls\\datasets\\",
                                  train=False, transform=torchvision.transforms.ToTensor())
TrainLoader = DataLoader(TrainDataset, batch_size=config.batch_size, shuffle=True)
TestLoader = DataLoader(TestDataset, batch_size=config.batch_size, shuffle=True)

# 加载网络
net = alexnet()

# 设置优化方式，定义损失函数
criterion = config.loss_func
optimizer = optim.SGD(net.parameters(), config.lr, momentum=0.9)


# 将打印信息保存至txt
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('train.txt')
print(time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime(time.time())))
print('------------------')
print('\n'.join(['%s:%s' % item for item in config.__dict__.items()]))
print('------------------')


# 训练
time_list = []
loss_list = []
start = time.clock()
time_list.append(start / 60)
for epoch in range(config.epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(TrainLoader):
        # get the inputs
        inputs = imgs
        # print(imgs.shape)
        labels = labels

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)


        # print(outputs.data.shape, outputs.data.item())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        loss_list.append(loss.item())
        running_loss += loss.item()
        if i % 10 == 9:
            print('Loss of epoch %-1s batch %-4s: %-10.9f' % (str(epoch + 1), str(i + 1), running_loss / 10))
            running_loss = 0.0
    time_list.append(time.clock() / 60 - sum(time_list))
    print('Epoch %-1s total run time : %-10.9f minutes' % (str(epoch + 1), time_list[epoch + 1]))

    if epoch % config.save_gap == (config.save_gap - 1):
        SaveWeight(net, weight_name = ("cifar10_alexnet_" + str(epoch + 1)))
        print("Saved the weight of epoch "+ str(epoch + 1))
        print('------------------')

print('Total training time : %-10.9f minutes' % (time.clock() / 60 - start))
x = range(0,len(loss_list))
y = loss_list
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x, y)
plt.ylabel("Loss")
plt.savefig("train_loss.png")
plt.show()