from model import alexnet
from runner.weight_save import LoadWeight


class WeightAnalysis():
    def __init__(self, weight_name):
        load_weight = self.loadweight(weight_name=weight_name)
        self.net = load_weight.net
        # model中所有参数名
        self.parameters = self.net.state_dict()
        # model中所有参数名及具体的值(包括通过继承得到的父类中的参数)
        self.para_iterator = self.net.named_parameters()

    def loadweight(self, weight_name):
        return LoadWeight(net=alexnet(), weight_name=weight_name)


if __name__ == "__main__":
    # test_weight = WeightAnalysis("cifar10_alexnet_5")
    # print(test_weight.parameters)      # 获得参数，可通过层名称索引获得权重，如['features.0.weight']
    # print(test_weight.para_iterator)   # 获得参数的迭代器
    """
    for para in test_weight.para_iterator:
        print("%-20s %-20s" % (para[0], para[1].shape))  # 打印层名称(self.name中的name+第n层中的n+weight/bias)，大小
                                                         # 第n层计算时，要加ReLU、池化、Dropout
    """