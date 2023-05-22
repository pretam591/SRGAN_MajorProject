from PyQt6.QtWidgets import QWidget, QLabel, QApplication, QPushButton, QRadioButton, QFileDialog
from PyQt6 import QtGui
import sys
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import *
from PIL import Image, ImageDraw, ImageFilter
from PIL.ImageQt import ImageQt
import numpy as np
from numpy import asarray
from copy import deepcopy
import torchvision.transforms as transforms
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Deep Neural Network Image Manipulator Application'
        self.setStyleSheet("background-color: lightblue;")
        self.left = 50 #xposition
        self.top = 50  #yposition
        self.width = 1250
        self.height = 650
        self.crop_image_resize = None
        self.model_path = None
        self.initUI()
        self.show()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height) #x,y,width,height
        
        self.label = QLabel('developed by Pretam Chandra',self)
        self.label.setGeometry(50, 50, 512, 512)
        self.label1 = QLabel('a 4th year Adamas University student',self)
        self.label1.setGeometry(650, 50, 512, 512)
        
        self.load_button = QPushButton("Load Image", self)
        self.load_button.setGeometry(50, 20, 100, 25)
        self.load_button.clicked.connect(self.load_image)
        
        self.sharp_button = QPushButton("Sharpen", self)
        self.sharp_button.setGeometry(850, 575, 90, 40)
        self.sharp_button.clicked.connect(self.sharpen_image)
        
        self.deep_sharp_button = QPushButton("Load Model", self)
        self.deep_sharp_button.setGeometry(950, 575, 90, 40)
        self.deep_sharp_button.clicked.connect(self.deep_sharpen_image)
        
        # creating a "2.5x" radio button
        self.radio_2_5x = QRadioButton('2.5x', self)
        self.radio_2_5x.setGeometry(180, 570, 120, 40)
        self.radio_2_5x.setStyleSheet("border: 1px solid; border-color:red;")
        self.radio_2_5x.clicked.connect(self.radio_2_5x_clicked)

        # creating a "10x" radio button
        self.radio_10x = QRadioButton('10x', self)
        self.radio_10x.setGeometry(350, 570, 120, 40)
        self.radio_10x.setStyleSheet("border: 1px solid; border-color:red;")
        self.radio_10x.clicked.connect(self.radio_10x_clicked)
        
    def radio_2_5x_clicked(self):
        print('2.5x clicked')
        self.rh = 512/2.5
        self.rw = 512/2.5
        self.model_path = None

    def radio_10x_clicked(self):
        print('10x clicked')
        self.rh = 512/10
        self.rw = 512/10
        self.model_path = None

    def load_image(self):
        # Loading the input image with PIL and resizing it to (512, 512)
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            image = Image.open(filename)
            print(image.size)
            pilImg = image.resize((512, 512))
             # Convert the PIL image to a QPixmap and set it as the input label's pixmap
            pixmap = QPixmap.fromImage(ImageQt(pilImg.convert('RGBA')))
            self.label.setPixmap(pixmap)
            
            self.ori_img = deepcopy(pilImg)
            self.backup_img = deepcopy(self.ori_img)
            
            self.radio_2_5x.setChecked(True)
            self.label.mousePressEvent = self.getPos
            return(self.label)
            
    def display_image(self, image, flag):
        # Displaying an image via converting it to a pixmap
        pixmap = QPixmap.fromImage(ImageQt(image.convert('RGBA')))

        if flag == 0:
            self.label.setPixmap(pixmap)
        if flag == 1:
            self.label1.setPixmap(pixmap)
            

    def sharpen_image(self):
        
        # Applying a sharp filter
        sharpened1 = self.crop_image_resize.filter(ImageFilter.SHARPEN);
        sharpened2 = sharpened1.filter(ImageFilter.SHARPEN);
        
        self.display_image(sharpened2, 1)
        
    def deep_sharpen_image(self):
        # Loading the neural network model with torch
        if self.model_path == None:
            self.model_path, _ = QFileDialog.getOpenFileName(self, 'Open Model', '', 'Model (*.pt)')
            print("welcome")
            model = torch.load('self.model_path', map_location=torch.device('cpu'))
            print("done3")
            print(model.input_shape)
            print(model.input_dtype)
        
        print("done4")
        # Defining a transform while converting PIL image to a Torch tensor
        transform = transforms.Compose([
            transforms.PILToTensor()
         ])
        img_tensor = transform(self.crop_image_resize).unsqueeze(0)
        print("done5")
        # Resizing the tensor to match the input size of our DL model
        resized_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        print("done6")
        # Passing the tensor through the model
        output_tensor = model(resized_tensor)
        print("done7")
        # Converting the output tensor back to an image
        output_img = transforms.ToPILImage()(output_tensor.squeeze().detach().cpu())
        print("done8")
        pixmap = QPixmap.fromImage(output_img)
        self.label1.setPixmap(pixmap)
        print("done9")
        #print(output_img)
        print("done10")
        # Convert the output PIL image to a QPixmap and set it as the output label's pixmap
        #self.display_image(output_image,1)
        print("done8")
        
        
    def getPos(self,event):
        cx, cy = event.pos().x(),event.pos().y()
        print(cx,cy) #prints the coordinates of the point clicked over the image
        start_point = (cx-int(self.rh/2), cy-int(self.rw/2)) #left topmost starting point of rectangle
        end_point = (cx+int(self.rh/2), cy+int(self.rw/2)) #right lowermost end-point of rectangle
        
        if cx > int(self.rh/2) and cx < (512-int(self.rh/2)) and cy > int(self.rh/2) and cy < (512-int(self.rh/2)): #safe-space creation
            #safe_coordinates = (102.4,102.4) -- (409.6,102.4)
                                #(102.4,409.6) -- (409.6,409.6)
            self.ori_img = deepcopy(self.backup_img)
            img = ImageDraw.Draw(self.ori_img)  
            img.rectangle([start_point , end_point], outline ="blue")
            
            self.display_image(self.ori_img,0)
        
            crop_image = self.ori_img.crop((start_point[0]+1, start_point[1]+1, end_point[0], end_point[1]))
            self.crop_image_resize = crop_image.resize((512, 512))
            self.display_image(self.crop_image_resize,1)
            
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec())