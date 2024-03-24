import os
from PIL import Image
import numpy as np


# 转换为灰度矩阵并添加标签
def read_png_to_gray_matrix(folder_path, processed_dataset):
    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # 拼接文件路径并读取
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            # 转换为灰度图像
            gray_matrix = image.convert('L')
            # 添加标签
            dic = {"label": label_generate(folder_path), "matrix": np.asarray(gray_matrix)}
            processed_dataset.append(dic)
    return processed_dataset


# 通过路径获取标签
def label_generate(path):
    if path == train_data_path[0] or path == test_data_path[0]:
        label = "angry"
        return label
    if path == train_data_path[1] or path == test_data_path[1]:
        label = "disgust"
        return label
    if path == train_data_path[2] or path == test_data_path[2]:
        label = "fear"
        return label
    if path == train_data_path[3] or path == test_data_path[3]:
        label = "happy"
        return label
    if path == train_data_path[4] or path == test_data_path[4]:
        label = "neutral"
        return label
    if path == train_data_path[5] or path == test_data_path[5]:
        label = "sad"
        return label
    if path == train_data_path[6] or path == test_data_path[6]:
        label = "surprise"
        return label
    else:
        raise ValueError("Invalid path")


# 训练集路径
train_data_path = []
path = 'original_data_set/train/angry'
train_data_path.append(path)
path = 'original_data_set/train/disgust'
train_data_path.append(path)
path = 'original_data_set/train/fear'
train_data_path.append(path)
path = 'original_data_set/train/happy'
train_data_path.append(path)
path = 'original_data_set/train/neutral'
train_data_path.append(path)
path = 'original_data_set/train/sad'
train_data_path.append(path)
path = 'original_data_set/train/surprise'
train_data_path.append(path)

# 测试集路径
test_data_path = []
path = 'original_data_set/test/angry'
test_data_path.append(path)
path = 'original_data_set/test/disgust'
test_data_path.append(path)
path = 'original_data_set/test/fear'
test_data_path.append(path)
path = 'original_data_set/test/happy'
test_data_path.append(path)
path = 'original_data_set/test/neutral'
test_data_path.append(path)
path = 'original_data_set/test/sad'
test_data_path.append(path)
path = 'original_data_set/test/surprise'
test_data_path.append(path)


# 处理训练集
def process_train_data(train_data_path):
    processed_dataset = []
    for path in train_data_path:
        processed_dataset = read_png_to_gray_matrix(path, processed_dataset)

    # 打乱训练集
    np.random.shuffle(processed_dataset)

    # 保存数据
    write_path = "processed_data_set/train/data.npy"
    np.save(write_path, processed_dataset, allow_pickle=True)

    # 输出测试,不用时改成False
    test = True
    if test:
        check = np.load(write_path, allow_pickle=True)
        print(check[:10])


# 处理测试集
def process_test_data(test_data_path):
    processed_dataset = []
    for path in test_data_path:
        processed_dataset = read_png_to_gray_matrix(path, processed_dataset)

    # 打乱训练集
    np.random.shuffle(processed_dataset)

    # 保存数据
    write_path = "processed_data_set/test/data.npy"
    np.save(write_path, processed_dataset, allow_pickle=True)

    # 输出测试,不用时改成False
    test = True
    if test:
        check = np.load(write_path, allow_pickle=True)
        print(check[:10])


process_train_data(train_data_path)
process_test_data(test_data_path)
