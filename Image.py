import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import *
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from typing import List
rgb_channels = ["red", "green", "blue"]

class Image:
    def __init__(self, image_arr = None):
        self.image = image_arr
        self.image_path = None
    
    def read_image(self, path):
        self.image_path = path    
        self.image = cv2.imread(self.image_path)
            
        # Convert BGR to RGB if image has 3 channels
        if self.is_RGB():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            print("RGB image read")
        else:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            print("Grayscale image read")    
    
    def is_RGB(self):
        """Checks if an image is RGB or 1-channel grayscale."""
        if self.image is None:
            raise ValueError("Image is not loaded.")
        
        # Check if the image has 3 channels (RGB)
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            return True  # RGB image
        elif len(self.image.shape) == 2:
            return False  # 1-channel grayscale image
        else:
            raise ValueError("Unexpected image format.")
    
    def get_bits_per_pixel(self):
        return self.image.itemsize *8
    
    
    def display_image(self):
        """Displays the image as a QGraphicsScene."""
        if self.image is None:
            raise ValueError("Image is not loaded.")

        # Ensure the image is either RGB or 1-channel grayscale
        if not self.is_RGB() and len(self.image.shape) != 2:
            raise ValueError("Unsupported image format. Only RGB or 1-channel grayscale images are allowed.")

        height, width = self.image.shape[:2]
        img_data = np.ascontiguousarray(self.image).tobytes()

        if self.is_RGB():  # RGB image
            bytes_per_line = 3 * width
            q_image = QImage(img_data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Grayscale image
            bytes_per_line = width
            q_image = QImage(img_data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Create a QGraphicsScene and add the pixmap
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        return scene
          

            
    def get_image(self):
        return np.copy(self.image)
    
    def get_cdf_canvas(self):
        return self.__cdf_canvas
    
    def rgb2gray(self):
        if self.is_RGB():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) 
            print("RGB image converted to gray")
        else: print("Image is already in grayscale")       