import os
import sys
from PyQt5 import QtWidgets, QtGui
# from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from mainUI import Ui_Form
from delay import *
# from neuralStyleTransfer import *
# from keras_preprocessing.image import load_img, img_to_array
# import numpy as np
# from keras.applications import vgg19
# from keras import backend as K
# from scipy.optimize import fmin_l_bfgs_b
# import imageio
# import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class mywindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("个性化衣物图案生成器")
        self.select_target_image_button.clicked.connect(self.openTargetImage)
        self.select_style_image_button.clicked.connect(self.openStyleImage)
        self.convert_button.clicked.connect(self.startConvert)

    # 选择目标图片上传
    def openTargetImage(self):
        global imgNamepath  # 这里为了方便别的地方引用图片路径，将其设置为全局变量
        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是默认打开的路径，第四个参数是需要的格式
        imgNamepath, imgType = QFileDialog.getOpenFileName(None, "选择图片",
                                                           os.getcwd(),
                                                           "All Files(*)")
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QtGui.QPixmap(imgNamepath).scaled(self.target_image.width(), self.target_image.height())
        # 在label控件上显示选择的图片
        self.target_image.setPixmap(img)
        # 显示所选图片的路径
        self.target_image_path.setText(imgNamepath)

    # 选择风格图片上传
    def openStyleImage(self):
        global imgNamepath  # 这里为了方便别的地方引用图片路径，将其设置为全局变量
        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是默认打开的路径，第四个参数是需要的格式
        imgNamepath, imgType = QFileDialog.getOpenFileName(None, "选择图片",
                                                           os.getcwd(),
                                                           "All Files(*)")
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QtGui.QPixmap(imgNamepath).scaled(self.style_image.width(), self.style_image.height())
        # 在label控件上显示选择的图片
        self.style_image.setPixmap(img)
        # 显示所选图片的路径
        self.style_image_path.setText(imgNamepath)

    # 生成素描图 多线程
    def startConvert(self):
        self.instruction.setText("正在转换中，请稍后......（根据电脑性能，大约会消耗(迁移程度×5~30)秒）")
        self.styleTransfer = StyleTransfer(self.target_image_path.text(), self.style_image_path.text(), self.iteration.text())
        self.styleTransfer.signal_over.connect(self.thread_style_transfer_over)
        self.styleTransfer.start()

    # 删除多线程
    def thread_style_transfer_over(self, str):
        print(str)
        self.instruction.setText("转换完成，图片已保存在可执行程序同级文件夹下（./result.jpg）")
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QtGui.QPixmap(os.getcwd() + "\\result.jpg").scaled(self.result_image.width(), self.result_image.height())
        # 在label控件上显示选择的图片
        self.result_image.setPixmap(img)
        del self.styleTransfer


#调用show
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())

# 为何如下一段代码运行后无法显示界面？
# class MyMainForm(QtWidgets.QWidget, Ui_Form):
#     def __int__(self):
#         super(MyMainForm, self).__init__()
#         self.setupUi(self)
#
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     myWin = MyMainForm()
#     myWin.show()
#
#     sys.exit(app.exec_())

# 如下一段代码可以显示界面，但未封装成类
# if __name__ == "__main__":
#  app = QtWidgets.QApplication(sys.argv)
#  widget = QtWidgets.QWidget()
#  ui = Ui_Form()
#  ui.setupUi(widget)
#  widget.show()
#  sys.exit(app.exec_())