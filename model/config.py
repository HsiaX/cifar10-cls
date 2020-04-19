import torch.nn as nn

__all__ = ['CifarConfig']

class CifarConfig():
    def __init__(self, num_classes = 10, lr = 0.001, loss_func = nn.CrossEntropyLoss(),
                 trained_model = "C:\\code\\cifar10-cls\\weights", save_gap = 1,
                 epoch = 5, batch_size = 32, preheating = False):
        self.num_classes = num_classes
        self.lr = lr
        self.loss_func = loss_func
        self.trained_model = trained_model
        self.save_gap = save_gap
        self.epoch = epoch
        self.batch_size = batch_size
        self.preheating = preheating


