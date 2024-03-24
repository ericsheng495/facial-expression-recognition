import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.optimizers import Adam
from keras.src.optimizers import SGD
from keras.layers import Dropout


# CNN模型
def cnn_model(input_shape, num_classes):
    model = models.Sequential()
    # （卷积层+池化层）*3
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    # 矩阵拉直
    model.add(layers.Flatten())
    # 全连接层
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    # dropout层
    model.add(Dropout(0.5))
    # 输出层
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# label to number
def label_generate(label):
    number = None
    if label == "angry":
        number = 0
    if label == "disgust":
        number = 1
    if label == "fear":
        number = 2
    if label == "happy":
        number = 3
    if label == "neutral":
        number = 4
    if label == "sad":
        number = 5
    if label == "surprise":
        number = 6
    return number


# 模型参数
input_shape = (48, 48, 1)
num_classes = 7

# 构建CNN模型
model = cnn_model(input_shape, num_classes)

# 读取数据, 路径需要修改
train_data_path = "D:/GT/2024 spring/CS 4641 ML/4641_project/processed_data_set/train/data.npy"
train_data = np.load(train_data_path, allow_pickle=True)
test_data_path = "D:/GT/2024 spring/CS 4641 ML/4641_project/processed_data_set/test/data.npy"
test_data = np.load(test_data_path, allow_pickle=True)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
image_matrix = np.array([item["matrix"] for item in train_data])
labels = np.array([label_generate(item['label']) for item in train_data])
test_image_matrix = np.array([item["matrix"] for item in test_data])
test_labels = np.array([label_generate(item['label']) for item in test_data])
history = model.fit(image_matrix, labels, epochs=30, batch_size=64, validation_data=(test_image_matrix, test_labels))

# 保存模型
model.save('model/trained_cnn_model.h5')
