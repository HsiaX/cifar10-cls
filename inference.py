import os
import sys
import time
import torch
import random
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
from runner import LoadWeight, PreHeating, Logger
from model import alexnet, CifarConfig

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 加载配置信息
config = CifarConfig()
# config.preheating = True

# 加载网络
LoadWeight = LoadWeight(net = alexnet(), weight_name = "cifar10_alexnet_5")
net = LoadWeight.net

# 模型预热
if config.preheating:
    PreHeating(net)

# 将打印信息保存至txt
path = os.path.abspath(os.path.dirname(__file__))
# type = sys.getfilesystemencoding()
sys.stdout = Logger('inference.txt')
print(time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime(time.time())))
print('------------------')
print('\n'.join(['%s:%s' % item for item in config.__dict__.items()]))
print('------------------')


# 随机选取指定文件夹下图片进行测试
ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
file_names = next(os.walk(IMAGE_DIR))[2]
img = Image.open(os.path.join(IMAGE_DIR, random.choice(file_names)))
input_img = img
if not (img.size == np.ones([32, 32]).shape):
    input_img = torchvision.transforms.functional.resize(input_img, (32, 32))

trans = torchvision.transforms.ToTensor()
input_img = trans(input_img)
input_img = input_img.view(1,3,32,32)


# 测试
start = time.clock()
output = net(input_img)
output = F.softmax(output, dim=1)
end = time.clock()

probability, predicted = torch.max(output.data, 1)
img_class = class_name[predicted.item()]

# 可视化
print('Category name : %-10s Inference time : %-10.9f seconds' % (str(img_class), (end - start)))
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.title(('%.2f probability to be %s' % (probability.item(), str(img_class))), y=-0.1)
plt.savefig("infer_result_1.png")
plt.show()
