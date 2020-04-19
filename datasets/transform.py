import os
import cv2
import json
import pickle

__all__ = ['unpickle', 'get_pkl_filenames', 'load_pkl_files',
           'ndarray2img', 'pkl2img', 'pkl2json_annotation', 'pkl2img_anno']

# 读pkl文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 获取指定路径下pkl文件名
def get_pkl_filenames(path):
    pkl_files = []
    filenames = os.listdir(path)
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.pkl':  # 目录下包含的.pkl文件
            pkl_files.append(filename)
    return pkl_files


# 返回指定路径下的pkl文件导出的dict
def load_pkl_files(path):
    pkl_dicts = []
    pkl_filenames = get_pkl_filenames(path)
    for pkl_filename in pkl_filenames:
        pkl_dicts.append(unpickle(path + pkl_filename))
    return pkl_dicts


def ndarray2img(b, g, r):
    img = cv2.merge([b, g, r])
    return img


# 读取pkl_dict中的图片数据并存入指定文件夹
def pkl2img(pkl_dicts, path):
    for i in range(1,6):
        for j in range(len(pkl_dicts[i][b'data'])):
            b = pkl_dicts[i][b'data'][j][2048:3072].reshape([32,32])
            g = pkl_dicts[i][b'data'][j][1024:2048].reshape([32,32])
            r = pkl_dicts[i][b'data'][j][0:1024].reshape([32,32])
            img_name = (pkl_dicts[i][b'filenames'][j]).decode()
            cv2.imwrite(path + "train_img\\" + img_name, ndarray2img(b, g, r))
        print("finish image transform of train batch " + str(i))

    for i in range(len(pkl_dicts[6][b'data'])):
        b = pkl_dicts[6][b'data'][2048:3072].reshape([32, 32])
        g = pkl_dicts[6][b'data'][1024:2048].reshape([32, 32])
        r = pkl_dicts[6][b'data'][0:1024].reshape([32, 32])
        img_name = (pkl_dicts[6][b'filenames'][i]).decode()
        cv2.imwrite(path + "test_img\\" + img_name, ndarray2img(b, g, r))
    print("finish image transform of test batch")


# 读取pkl文件并保存标注
def pkl2json_annotation(pkl_dicts, path):
    json_dict_train = {"info": [], "images": [], "categories": []}
    json_dict_train['info'] = "This is the annotation of Cifar10 train dataset."
    json_dict_test = {"info": [], "images": [], "categories": []}
    json_dict_test['info'] = "This is the annotation of Cifar10 test dataset."
    for i in range(1, 6):
        for j in range(len(pkl_dicts[i][b'data'])):
            img_name = (pkl_dicts[i][b'filenames'][j]).decode()
            # img_cls = label_names[pkl_dicts[i][b'labels'][j]]
            # one_hot_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 生成全0矩阵
            # one_hot_label[pkl_dicts[i][b'labels'][j]] = 1
            img_cls = pkl_dicts[i][b'labels'][j]
            json_dict_train['images'].append(img_name)
            json_dict_train['categories'].append(img_cls)
        print("finish annatation transform of train batch " + str(i))
    json.dump(json_dict_train, open(path + "train_annotations.json",'w'))
    print("finish annatation transform of train batch")

    for i in range(len(pkl_dicts[6][b'data'])):
        img_name = (pkl_dicts[6][b'filenames'][i]).decode()
        # img_cls = label_names[pkl_dicts[6][b'labels'][i]]
        # one_hot_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 生成全0矩阵
        # one_hot_label[pkl_dicts[6][b'labels'][i]] = 1
        img_cls = pkl_dicts[6][b'labels'][i]
        json_dict_test['images'].append(img_name)
        json_dict_test['categories'].append(img_cls)
    print("finish annatation transform of test batch")
    json.dump(json_dict_test, open(path + "test_annotations.json",'w'))


# 读取pkl文件，将图片保存到标注名文件夹下
def pkl2img_anno(pkl_dicts, label_names, path):
    for i in range(1,6):
        for j in range(len(pkl_dicts[i][b'data'])):
            b = pkl_dicts[i][b'data'][j][2048:3072].reshape([32,32])
            g = pkl_dicts[i][b'data'][j][1024:2048].reshape([32,32])
            r = pkl_dicts[i][b'data'][j][0:1024].reshape([32,32])
            img_name = (pkl_dicts[i][b'filenames'][j]).decode()
            cv2.imwrite(path + "train\\" + str(label_names[pkl_dicts[i][b'labels'][j]]) + "\\" + img_name, ndarray2img(b, g, r))
        print("finish train batch " + str(i))

    for i in range(len(pkl_dicts[6][b'data'])):
        b = pkl_dicts[b'data'][6][2048:3072].reshape([32, 32])
        g = pkl_dicts[b'data'][6][1024:2048].reshape([32, 32])
        r = pkl_dicts[b'data'][6][0:1024].reshape([32, 32])
        img_name = (pkl_dicts[b'filenames'][6][i]).decode()
        cv2.imwrite(path + "test\\" + str(label_names[pkl_dicts[b'labels'][6][i]]) + "\\" + img_name, ndarray2img(b, g, r))
    print("finish test batch ")


if __name__ == "__main__":

    # 将目录下所有pkl分别读成dict，并组合成一个list
    pkl_dicts = load_pkl_files("C:\\code\\cifar10-cls\\cifar10-oridata\\")
    # print(pkl_dicts[0][b'label_names'])
    label_names = []  # 类别名
    for i in range(len(pkl_dicts[0][b'label_names'])):
        label_names.append((pkl_dicts[0][b'label_names'][i]).decode())

    # 将图片保存至对应标注文件夹下
    # pkl2img_anno(pkl_dicts, label_names, "C:\\code\\cifar10-cls\\datasets\\")

    # 保存图片
    # pkl2img(pkl_dicts, "C:\\code\\cifar10-cls\\datasets\\")

    # 保存标注
    # pkl2json_annotation(pkl_dicts, "C:\\code\\cifar10-cls\\datasets\\")
