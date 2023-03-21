from PyQt5.QtCore import QThread, pyqtSignal

from neuralStyleTransfer import *


class StyleTransfer(QThread):
    # 使用信号和UI主线程通讯
    signal_over = pyqtSignal(str)

    def __init__(self, target_image_path, style_image_path, iteration, parent=None):
        super(StyleTransfer, self).__init__(parent)
        self.target_image_path = target_image_path
        self.style_image_path = style_image_path
        self.iteration = iteration

    def run(self):
        neuralStyleTransfer(target_image_path=str(self.target_image_path),
                            style_reference_image_path=str(self.style_image_path),
                            iterations=int(self.iteration))
        self.signal_over.emit("Transfer finished!")
        return
