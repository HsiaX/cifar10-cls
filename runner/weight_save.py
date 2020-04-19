import torch

__all__ = ['SaveWeight', 'LoadWeight']


class SaveWeight():
    def __init__(self, net, path = "C:\\code\\cifar10-cls\\weights\\", save_net = False, weight_name = "net.pkl"):
        self.net = net
        self.path = path
        self.weight_name = weight_name
        if save_net:
            torch.save(self.net, self.path + self.weight_name)
        else:
            torch.save(self.net.state_dict(), self.path + self.weight_name)


class LoadWeight():
    def __init__(self, net, path = "C:\\code\\cifar10-cls\\weights\\", load_net = False, weight_name = "net.pkl"):
        self.path = path
        self.weight_name = weight_name
        if load_net:
            self.net = torch.load(self.path + self.weight_name)
        else:
            self.net = net
            self.net.load_state_dict(torch.load(self.path + self.weight_name))


class LoadWeight_part():
    # 接续训练
    pass