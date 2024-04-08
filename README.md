# handwritten_digit_recognition
学习与实现  https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/  
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/5e6ef416-3604-4553-baa4-bf4ee4192120)  
## 一、目录结构  
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/7c50d98a-4fe2-4dfa-882b-94feea0362f9)  
## 二、文件说明 
    1.train.py：训练模型，产生mnist.h5文件  
    2.gui.py：展示图形界面，可以进行模型微调  
## 三、笔记  
### os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
这行代码是用来设置环境变量 `KMP_DUPLICATE_LIB_OK` 的值为 `"TRUE"`。这个环境变量通常与 Intel MKL（Math Kernel Library）相关的问题有关，特别是在使用多线程时可能会出现的问题。  
解释一下这个设置的作用：  
1. KMP_DUPLICATE_LIB_OK：  
 - 这个环境变量是用来控制是否允许 Intel MKL 库加载多次的。  
（Intel MKL 是一个强大的数学函数库，旨在优化数值计算和科学计算应用程序的性能，特别是针对英特尔处理器进行了优化，为开发人员提供了高效的数学函数和算法支持。）  
- 当 `KMP_DUPLICATE_LIB_OK` 的值为 `"TRUE"` 时，表示允许 Intel MKL 库被加载多次。  
- 如果不设置或者设置为其他值（如 `"FALSE"`），则可能会导致某些问题，特别是在多线程环境下。  
2. 为什么要设置为 "TRUE"：  
- 在某些情况下，特别是在使用了多线程的深度学习框架（如 TensorFlow 或 PyTorch）时，可能会遇到 Intel MKL 库重复加载的问题。  
- 通过将 `KMP_DUPLICATE_LIB_OK` 设置为 `"TRUE"`，可以解决一些由于多线程环境下 Intel MKL 库重复加载导致的问题，从而提高程序的稳定性和性能。  
总的来说，设置 `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"` 可以帮助解决某些与 Intel MKL 库重复加载相关的问题，特别是在使用深度学习框架时可能会遇到的一些问题。  
***
### (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    Keras 中加载 MNIST 数据集，函数返回一个包含训练集和测试集的元组。返回时已经将数据集划分为训练集和测试集。  
    使用 Python 的元组解构（tuple unpacking）的方式，将 mnist.load_data() 返回的元组中的训练集和测试集数据分别赋值给 (x_train, y_train) 和 (x_test, y_test) 这两个变量。  
    x_train 和 x_test 是图像数据，通常是一个 28x28 的灰度图像，表示手写数字的图像。  
    y_train 和 y_test 是标签数据，表示对应图像的数字标签（0 到 9）。  
***  
### reshape 
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)：这行代码对训练集中的图像数据进行了形状调整。  
    x_train 是训练集中的图像数据，原始形状为 (样本数, 28, 28)，表示每张图像是一个 28x28 的灰度图像。  
    通过 reshape 操作，将每张图像的形状调整为 (28, 28, 1)，其中最后一个维度是 1，表示这是一个单通道（灰度）图像。  
***
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/ae699799-af5e-4c99-a085-ba66a2e1f99c)  
解决方法：科学上网一下  
***
### keras.utils.to_categorical函数  
    （ https://blog.csdn.net/qian2213762498/article/details/86584335 ）  
    • 把类别标签转换为onehot编码（categorical就是类别标签的意思）， 而onehot编码是一种方便计算机处理的二元编码。  
    • to_categorical(y, num_classes=None, dtype='float32')y为int数组，num_classes为标签类别总数，大于max(y)（标签从0开始的）  
    • 返回：如果num_classes=None，返回len(y) * [max(y)+1]（维度，m*n表示m行n列矩阵，下同），否则为len(y) * num_classes  
    • 具体功能：（来源： https://www.cnblogs.com/MilkAndPudding/p/16055731.html ）  
    原来类别向量中的每个值都转换为矩阵里的一个行向量，从左到右依次是0,1,2，...8个类别。2表示为[0. 0. 1. 0. 0. 0. 0. 0. 0.]，只有第3个为1，作为有效位，其余全部为0。  
    • one_hot encoding(独热编码)介绍  
    独热编码又称为一位有效位编码，上边代码例子中其实就是将类别向量转换为独热编码的类别矩阵  
***
### 图像处理--像素（在大多数情况下，可以将像素值的范围理解为在 0 到 255 之间） 
图像的像素值可以通过编程库如OpenCV等进行读取和修改。以下是具体的方法： 
1. 读取像素值：  
 - 使用OpenCV，你可以通过指定图像矩阵的行列索引来读取某个特定像素的值。如果是灰度图像，你会得到一个单独的数值表示灰度级别；对于彩色图像，通常会得到一个包含三个数值的元组或数组，分别代表红色、绿色和蓝色的强度值。  
 - 使用`cv2.imread()`函数读取图片，并使用`img[row, column]`的方式来获取像素值。例如`p = img[100, 200]`会获取位于第100行第200列的像素值。  
 - 对于使用Numpy的情况，可以使用`img.item(row, column, channel)`的方式读取单个像素值。  
2. 修改像素值：  
 - 可以通过指定像素位置并赋予新的值来修改像素。  
例如，在OpenCV中，你可以使用`img[row, column] = value`来设置单个像素的值。如果`value`是一个元组，比如`(0, 0, 255)`，这会将该像素设置为红色。  
 - 若要修改一系列像素，可以使用切片语法。  
例如，`img[100:300, 150:350] = (0, 0, 255)`会将图像中从第100行到第300行、第150列到第350列的所有像素设置为红色。  
```python
import cv2
# 读取图像
image = cv2.imread('img.png')
# 获取图像的宽度、高度和通道数
height, width, channels = image.shape
print (height, width, channels)#1024 1024 3
# 获取像素值范围
min_pixel_value = image.min()
max_pixel_value = image.max()
print(f"最小像素值：{min_pixel_value}")#0
print(f"最大像素值：{max_pixel_value}")#255
# 获取特定位置处的像素值
pixel_value = image[100, 50]
print('pixel_value'+str(pixel_value))#pixel_value[0 0 0]
#转灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 获取图像的宽度、高度(灰度图像是以单通道的形式表示的，因此在获取灰度图像的形状时，不需要指定通道数。)
height, width= gray_image.shape
print (height, width)
# 获取像素值范围
min_pixel_value = gray_image.min()
max_pixel_value = gray_image.max()
print(f"最小像素值：{min_pixel_value}")
print(f"最大像素值：{max_pixel_value}")
```
***
### 网络结构介绍  
conv2D: 卷积神经网络层，参数包括：  
1. filters: 层深度（纵向），一般来说前期数据减少，后期数量逐渐增加，建议选择2N作为深度，比如说：[32,64,128] => [256,512,1024]；  
2. kernel_size: 决定了2D卷积窗口的宽度和高度，一般设置为( 1 × 1 ) ，( 3 × 3 )，( 5 × 5 )，( 7 × 7 )  
3. activation：激活函数，可选择为：sigmoid,tanh,relu等  
4. MaxPooling2D: 池化层，本质上是采样，对输入的数据进行压缩，一般用在卷积层后，加快神经网络的训练速度。没有需要学习的参数，数据降维，用来防止过拟合现象。  
5. Dropout：防过拟合层，在训练时，忽略一定数量的特征检测器，用来增加稀疏性，用伯努利分布（0-1分布）B(1,p)来随机忽略特征数量，输入参数为p的大小  
6. Flatten：将多维输入数据一维化，用在卷积层到全连接层的过渡，减少参数的使用量，避免过拟合现象，无参。  
7. Dense：全连接层，将特征非线性变化映射到输出空间上。  
***
### 训练模型 model.fit   
在搭建完成后，将数据送入模型进行训练  
○ x：训练数据输入；  
○ y：训练数据输出；  
○ batch_size： batch样本数量，即训练一次网络所用的样本数；  
○ epochs：迭代次数，即全部样本数据将被“轮”多少次，轮完训练停止；  
○ verbose：可选训练过程中信息是否输出参数，0表示不输出信息，1表示显示进度条(一般默认为1)，2表示每个epoch输出一行记录；  
○ valdation_data：验证数据集。  
返回值hist：包含训练过程中的历史信息，包括损失和准确率等指标的变化情况。  
使用`hist`变量可以帮助你获取模型训练过程中的历史信息，例如损失值和准确率的变化情况。通常，`hist`是一个字典，其中包含了训练过程中的各种指标值。你可以使用这些信息进行可视化、分析和调试。  
下面是一些示例代码，展示了如何使用`hist`中的信息：  
1. **获取训练过程中的损失值和准确率**：  
```python
train_loss = hist.history['loss']
train_accuracy = hist.history['accuracy']
```
2. **获取验证集上的损失值和准确率**：  
```python
val_loss = hist.history['val_loss']
val_accuracy = hist.history['val_accuracy']
```
3. **绘制训练过程中的损失曲线和准确率曲线**：  
```python
import matplotlib.pyplot as plt
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
```
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/51e816d4-8ddc-4451-938c-638bba735971)![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/1fccc555-274d-4ef8-80b0-fb24fbcb3037)  
Test loss: 0.26256537437438965  
Test accuracy: 0.9240999817848206  
通过上述示例，你可以利用`hist`中的信息对模型的训练过程进行可视化展示，从而更好地了解模型的训练情况和性能表现。这些信息有助于调整模型架构、优化超参数以及改进模型训练策略。  
***
### 识别结果  
原来的代码中是白底黑字，识别出来的效果堪忧（不过怎么感觉别人的结果挺好的），但是模型准确度应该没有问题，于是更改了一些。  
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/ebf2afd5-13c7-4fee-963d-51f55d664fe5)   
1.图像颜色反转  
```python	
from PIL import Image
#打开图像文件
image = Image.open("input.jpg")
#将图像转换为黑白模式
image = image.convert("L")
# 反转图像
image = Imageops.invert(image)
#保存反转后的图像
image.save("output.jpg")
```
2.直接把画布颜色改变（变成黑底白字）  
***
### 显示器缩放125%导致的画布捕捉不正确问题  
当显示器缩放为125%时，会导致屏幕上的坐标与实际像素坐标不完全匹配，从而影响截图的位置。  
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/8f1be1fc-7b9a-4a24-b11d-8403631ccab2)   
可以根据显示器缩放比例来调整捕捉画布的位置，在计算捕捉区域的坐标时，将每个坐标乘以1.25（即125%的缩放比例），以适应显示器的缩放比例。  
将高亮的那一行，代替为接下来几行即可  
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/541f9acb-0799-47f3-b994-6be95ebccb9c)  
***
将捕捉到的图片送入模型微调，再次预测可以看出结果有所不同:  
![image](https://github.com/SweetThreee/handwritten_digit_recognition/assets/107618206/90cd25c9-d174-46be-af02-c59b32eb1a8d) 


