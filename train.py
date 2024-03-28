import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

from keras.utils import plot_model
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"#允许 Intel MKL 库被加载多次
#函数返回一个包含训练集和测试集的元组
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)训练数据集维度为：60000 × 28 × 28 ，测试数据集维度为：10000 × 28 × 28
"""
数据预处理Preprocess the data
"""
#将图像数据转换为神经网络输入，图像大小60000 × 28 × 28 ，输出大小为60000 × 28 × 28 × 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)#每个输入样本是一个 28x28 的单通道灰度图像

num_classes = 10 #定义了数据集中的类别数量。在 MNIST 数据集中，共有 10 个类别，分别对应数字 0 到 9。
y_train = keras.utils.to_categorical(y_train, num_classes)#one-hot 编码转换，将其转换为一个二进制矩阵表示。
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')#通过astype()方法强制转换数据的类型
x_test = x_test.astype('float32')
x_train /= 255 #数据归一化：将像素值从整数范围 [0, 255] 缩放到 [0, 1] 的浮点数范围
x_test /= 255
print('x_train shape:', x_train.shape)# x_train shape: (60000, 28, 28, 1)
print(x_train.shape[0], 'train samples')# 60000 train samples
print(x_test.shape[0], 'test samples')# 10000 test samples

"""
创建卷积神经网络模型
"""
batch_size = 128
num_classes = 10
epochs = 50

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))#Dropout随机失活层（防止过拟合）
model.add(Flatten())
model.add(Dense(256, activation='relu'))#Dense密集层（全连接FC层，在Keras层中FC层被写作Dense层）
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#编译神经网络结构（Adadelta是一种优化器，用于在深度学习模型中更新权重以最小化损失函数）
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
plot_model(model, to_file='emotion_model.png', show_shapes=True)
"""
训练模型 model.fit：在搭建完成后，将数据送入模型进行训练
    x：训练数据输入；
    y：训练数据输出；
    batch_size： batch样本数量，即训练一次网络所用的样本数；
    epochs：迭代次数，即全部样本数据将被“轮”多少次，轮完训练停止；
    verbose：可选训练过程中信息是否输出参数，0表示不输出信息，1表示显示进度条(一般默认为1)，2表示每个epoch输出一行记录；
    valdation_data：验证数据集。
"""
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
model.save('mnist.h5')
train_loss = hist.history['loss']#获取训练过程中的损失值和准确率
train_accuracy = hist.history['accuracy']
val_loss = hist.history['val_loss']#获取验证集上的损失值和准确率
val_accuracy = hist.history['val_accuracy']
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()
plt.figure()
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()

"""
评价网络，返回值是一个浮点数，表示损失值和评估指标值，输入参数为测试数据，verbose表示测试过程中信息是否输出参数，同样verbose=0表示不输出测试信息。
"""
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
