import torch


__all__ = ['PreHeating']


class PreHeating():
    def __init__(self, net):
        self.net = net
        pre_img = torch.randn([3,3,32,32])
        pre_label = torch.tensor([1, 2, 3])
        pre_loader = [pre_img, pre_label]

        for i in range(3):
            # get the inputs
            img = pre_loader[0][i].view(1,3,32,32)
            # print(imgs.shape)
            label = pre_loader[1][i]
            outputs = net(img)