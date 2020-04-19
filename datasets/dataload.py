import cv2
import json
import torchvision
from torch.utils.data import Dataset, DataLoader

__all__ = ['Cifar10Dataset']

class Cifar10Dataset(Dataset):
    def __init__(self, img_path, json_path, train=True, transform=None):
        self.transform = transform
        train_dict = open(json_path + "train_annotations.json")
        test_dict = open(json_path + "test_annotations.json")
        train_annos = json.load(train_dict)
        test_annos = json.load(test_dict)
        train_path = img_path + "train_img\\"
        test_path = img_path + "test_img\\"

        if train:
            self.image_path = train_path
            self.annos = train_annos
        else:
            self.image_path = test_path
            self.annos = test_annos

        # 对应顺序保存
        self.image_list = []
        for i in range(len(self.annos["categories"])):
            self.image_list.append(self.annos["images"][i])

    def __getitem__(self, index):
        img_name = self.image_path + self.image_list[index]
        # label = torch.tensor(self.annos["categories"][index])
        label = self.annos["categories"][index]
        img = cv2.imread(img_name)

        if self.transform:
            try:
                img = self.transform(img)
            except:
                print("Cannot transform image: {}".format(img_name))

        return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    # 数据集实例化
    TrainDataset = Cifar10Dataset(img_path = "C:\\code\\cifar10-cls\\datasets\\",
                                  json_path = "C:\\code\\cifar10-cls\\datasets\\",
                                  train=True, transform=torchvision.transforms.ToTensor())
    TestDataset = Cifar10Dataset(img_path="C:\\code\\cifar10-cls\\datasets\\",
                                  json_path="C:\\code\\cifar10-cls\\datasets\\",
                                  train=False, transform=torchvision.transforms.ToTensor())
    TrainLoader = DataLoader(TrainDataset, batch_size=64, shuffle=True)
    TestLoader = DataLoader(TestDataset, batch_size=64, shuffle=True)

""" 测试
    for batch_num, (img, label) in enumerate(TrainLoader):
        if batch_num < 5:
            print(img.shape)
            imgs = torchvision.utils.make_grid(img)
            print(imgs.shape)
            print(label)
            imgs = np.transpose(imgs, (1,2,0))    # 将CxHxW转换为HxWxC，用plt显示
            print(imgs.shape)
            plt.imshow(imgs)
            plt.show()
"""