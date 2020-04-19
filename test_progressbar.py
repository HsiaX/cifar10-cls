import os
import sys
import time
import torch
import numpy as np
import torchvision
from visualization import track_progress, track_iter_progress, ProgressBar
from runner import LoadWeight, PreHeating, Logger
from model import alexnet, CifarConfig
from datasets import Cifar10Dataset
from torch.utils.data import DataLoader

# 加载配置信息
config = CifarConfig()
config.preheating = True

# 加载数据集
TrainDataset = Cifar10Dataset(img_path="C:\\code\\cifar10-cls\\datasets\\",
                              json_path="C:\\code\\cifar10-cls\\datasets\\",
                              train=True, transform=torchvision.transforms.ToTensor())
TestDataset = Cifar10Dataset(img_path="C:\\code\\cifar10-cls\\datasets\\",
                             json_path="C:\\code\\cifar10-cls\\datasets\\",
                             train=False, transform=torchvision.transforms.ToTensor())
TrainLoader = DataLoader(TrainDataset, batch_size=config.batch_size, shuffle=True)
TestLoader = DataLoader(TestDataset, batch_size=config.batch_size, shuffle=True)

# 加载网络，默认权重在weights文件夹下
LoadWeight = LoadWeight(net=alexnet(), weight_name="cifar10_alexnet_5")
net = LoadWeight.net

# 模型预热
if config.preheating:
    PreHeating(net)

# 将打印信息保存至txt
path = os.path.abspath(os.path.dirname(__file__))
# type = sys.getfilesystemencoding()
sys.stdout = Logger('test.txt')
print(time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime(time.time())))
print('------------------')
print('\n'.join(['%s:%s' % item for item in config.__dict__.items()]))
print('------------------')


def test(correct, total, accuracy, time_list):
    # track_progress(func=int, tasks=(time_list, 6)) # 放此处不更新
    for epoch in range(config.epoch):
        # print(isinstance(range(config.epoch), Iterable))
        # track_progress(func = len, tasks = (time_list, 5))
        for i, (imgs, labels) in enumerate(TestLoader):
            # get the inputs
            inputs = imgs
            # print(imgs.shape)
            labels = labels

            outputs = net(imgs)
            probability, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy.append(100 * correct / total)
        time_list.append(time.clock() / 60 - sum(time_list))
        print("epoch: %-5s accuracy: %-10.2f test time: %-10.9f minutes" % (
        str(epoch + 1), accuracy[epoch], time_list[epoch + 1]))
        # prog_bar.update()
        # track_iter_progress(tasks=(time_list, 6))
        track_progress(func=int, tasks=(time_list, 6))
    print('------------------')
    print("Average accuracy: %-7.2f Average test time: %-10.9f minutes" % (
    np.float(np.mean(accuracy)), np.float(np.mean(time_list))))

if __name__ == "__main__":
    # 测试
    correct = 0
    total = 0
    accuracy = []
    time_list = []
    start = time.clock()
    time_list.append(start)
    # track_progress(func=int, tasks=time_list)
    # prog_bar = ProgressBar(5)
    track_progress(func=int, tasks=(time_list, 6))
    test(correct, total, accuracy, time_list)