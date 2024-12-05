import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QPushButton,QLabel
from PyQt5.QtGui import QPixmap

from inference import getFFHQ
from inference1 import getCeleba


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300,300,600,450)
        self.setWindowTitle('U-Net GAN Image Generation')

        self.btn1 = QPushButton("FFHQ",self)
        self.btn2 = QPushButton("CelebA",self)
        self.btn1.setGeometry(100,150-25,100,50)
        self.btn2.setGeometry(100,300-25,100,50)
        self.btn1.clicked.connect(self.clickbtn1)
        self.btn2.clicked.connect(self.clickbtn2)


        self.exampleImg = QLabel(self)
        self.exampleImg.setGeometry(300,100,256,256)
        self.examplePixmap = QPixmap('14.png')
        self.exampleImg.setPixmap(self.examplePixmap)

        self.tips = QLabel(self)
        self.tips.setText("生成图片")
        self.tips.setGeometry(350,360,200,self.tips.height())

        self.show()

    def clickbtn1(self):
        self.FFHQpth = getFFHQ()
        self.tips.setText('FFHQ 生成图片')
        self.exampleImg.setPixmap(QPixmap(self.FFHQpth))
    def clickbtn2(self):
        self.Celebapth = getCeleba()
        self.tips.setText('CelebA 生成图片')
        self.exampleImg.setPixmap(QPixmap(self.Celebapth))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWindow()
    sys.exit(app.exec_())