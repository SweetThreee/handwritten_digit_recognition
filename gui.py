import cv2
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui #win32gui库，用于获取窗口信息。
from PIL import ImageGrab, Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt

'''
首先将图像调整为28x28像素大小，然后将RGB图像转换为灰度图像。
接下来，将图像转换为NumPy数组并进行归一化处理。
最后，使用Keras模型进行预测，并返回预测结果中概率最高的类别和对应的概率值。
'''
#输入是图像，对图像进行预处理后使用模型进行预测，返回预测的数字和置信度
def predict_digit(img):
    img = img.resize((28, 28)) #resize image to 28x28 pixels
    img = img.convert('L')    #convert rgb to grayscale
    img=ImageOps.invert(img)
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)#reshaping to support our model input and normalizing
    img = img/255.0
    # print(img)
    print(model.predict([img]))
    res = model.predict([img])[0]#predicting the class
    return np.argmax(res), max(res)
#创建一个Tkinter应用程序的类App
class App(tk.Tk):
    def __init__(self):#初始化函数，设置窗口中的组件。
        tk.Tk.__init__(self)
        self.x = self.y = 0
        '''
        在这个特定的上下文中，self.x和self.y被用来跟踪鼠标的坐标位置。
        通过将它们初始化为0，可以确保在应用程序启动时，鼠标的初始位置被设置为(0, 0)。
        随后，在用户绘制数字时，这两个变量会根据鼠标事件的发生而更新，用于确定绘制的笔画位置。
        '''
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross") #创建一个画布用于绘制数字
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))#创建一个标签用于显示识别结果。
        self.classify_btn = tk.Button(self, text = "Recognise", command =self.classify_handwriting)#创建一个按钮用于触发识别操作。
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)#创建一个按钮用于清除画布内容。
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)#绑定鼠标左键拖动事件到draw_lines函数。

    def clear_all(self):# 清除画布上的所有内容。
        self.canvas.delete("all")

    def classify_handwriting(self):#获取画布区域的截图，调用predict_digit函数进行识别，并更新标签显示结果。
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        print(rect)
        im = ImageGrab.grab(rect)
        plt.imshow(im)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):#绘制画布上的数字和实现鼠标绘制功能
        self.x = event.x
        self.y = event.y
        r=8
        '''
        这行代码是在Tkinter的画布上创建一个椭圆（oval），用来表示用户在画布上绘制的手写数字的笔画。下面是这行代码中各个参数的含义：
            self.canvas: 表示在Tkinter应用程序中创建的画布对象。
            create_oval(): 这是Tkinter画布对象的方法，用于创建一个椭圆形。
            self.x - r, self.y - r, self.x + r, self.y + r: 这四个参数分别表示椭圆的左上角和右下角的坐标，即椭圆的外接矩形。self.x和self.y表示椭圆的中心点坐标，r表示椭圆的半径。
            fill='black': 这个参数表示椭圆的填充颜色为黑色。椭圆将以黑色填充，表示用户绘制的手写数字的笔画。
        通过这行代码，每次用户在画布上拖动鼠标时，都会在鼠标位置处创建一个黑色填充的椭圆，从而实现了用户在画布上绘制手写数字的功能。
        '''
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black',outline='black')

if __name__ == '__main__':
    model = load_model('mnist.h5')
    app = App() # 创建App类的实例。
    mainloop() # 运行Tkinter的主事件循环，启动应用程序。
