import os
import sys
import time
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from runner import SaveWeight, LoadWeight, SaveWeight_checkpoint, LoadWeight_checkpoint, Logger
from model import alexnet, CifarConfig, alexnet_lightweight
from datasets import Cifar10Dataset
from torch.utils.data import DataLoader


# 加载配置信息，修改默认参数
config = CifarConfig()
config.lr = 0.01
config.batch_size = 32
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


# 加载网络，若断续重连则确定训练初始迭代数
# net = alexnet()
net = alexnet_lightweight()   # 新的轻量化网络
start_epoch = 0
if os.listdir('C:\\code\\cifar10-cls\\weights\\checkpoints\\'):
    filenames = []
    file_ctime = []
    for filename in os.listdir('C:\\code\\cifar10-cls\\weights\\checkpoints\\'):
        filenames.append(filename)
        file_ctime.append(os.path.getctime('C:\\code\\cifar10-cls\\weights\\checkpoints\\' + filename))
    index = file_ctime.index(max(file_ctime))
    checkpoint_filename = filenames[index]          # 获得最后生成的checkpoint模型
    LoadWeight_c = LoadWeight_checkpoint(net = net, weight_name = checkpoint_filename)
    net = LoadWeight_c.net
    start_epoch = LoadWeight_c.epoch_num + 1


# 设置优化方式，定义损失函数
criterion = config.loss_func
# optimizer = optim.SGD(net.parameters(), config.lr)
optimizer = optim.SGD(net.parameters(), config.lr, momentum=0.9)
# optimizer = torch.optim.RMSprop(net.parameters(), config.lr, alpha=0.9)
# optimizer = torch.optim.Adam(net.parameters(), config.lr, betas=(0.9, 0.99))
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=0.80)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)


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
lr_list = []
start = time.clock()
time_list.append(start / 60)
for epoch in range(start_epoch, config.epoch):  # loop over the dataset multiple times
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
        if i % 100 == 99:            # 每隔100batch更新学习率
            scheduler.step()
            lr_list.append(scheduler.get_lr()[0])
    time_list.append(time.clock() / 60 - sum(time_list))
    print('Epoch: %-5s learning rate: %-9.4f total run time: %-10.9f minutes' % (str(epoch + 1), lr_list[len(lr_list) - 1], time_list[len(time_list) - 1]))

    if epoch % config.save_gap == (config.save_gap - 1):
        checkpoint = {}
        checkpoint = {'epoch': epoch, 'model': net.state_dict()}    # 建立checkpoint还原点，屏蔽该句则不建立
        if checkpoint:
            SaveWeight_checkpoint(checkpoint = checkpoint, weight_name = ("cifar10_checkpoint_" + str(epoch + 1)))
        else:
            SaveWeight(net = net, weight_name = ("cifar10_alexnet_" + str(epoch + 1)))
        print("Saved the weight of epoch "+ str(epoch + 1))
        print('------------------')

print('Total training time : %-10.9f minutes' % (time.clock() / 60 - start))
x1 = range(0,len(loss_list))
y1 = loss_list
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x1, y1)
plt.ylabel("Loss")
plt.savefig("train_loss.png")

x2 = range(0,len(lr_list))
y2 = lr_list
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x2, y2)
plt.ylabel("Learning rate")
plt.savefig("learning_rate.png")
plt.show()