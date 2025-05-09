from PyQt5 import QtWidgets, QtCore, uic
import sys
from PyQt5.QtGui import *
import numpy as np
import cv2
from sklearn.metrics.pairwise import euclidean_distances
import os

from Image import Image
import feature_extraction
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('gui.ui', self)
    
        # buttons connection
        self.upload_button.clicked.connect(self.upload_image)
        self.face_detection_button.clicked.connect(self.detect_faces)
        self.face_recognitio_button.clicked.connect(self.recognize_face)
        
        # load faces from data set
        X, self.y = feature_extraction.load_faces()
        self.eigenvectors, self.eigenvalues, self.X_transformed, self.mean = feature_extraction.train_faces(X)
        print("Training DONEEEE")
        
    def upload_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        
        if self.file_path:
            self.image = Image()
            self.image.read_image(self.file_path)
            scene = self.image.display_image()
            self.original_image_arr = np.copy(self.image.image)  
            self.output_graphicsView.setScene(scene)
            self.output_graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                        
                        
    def detect_faces(self):
        # Convert to grayscale (do NOT overwrite the original image)
        gray = cv2.cvtColor(self.image.image, cv2.COLOR_RGB2GRAY)

        # Load Haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles on a copy to avoid modifying original
        output_img = self.image.image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update the displayed image (temporarily replacing self.image.image for display)
        self.image.image = output_img
        scene = self.image.display_image()
        self.output_graphicsView.setScene(scene)
        self.output_graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
        
    def recognize_face(self):
        # Convert to grayscale (do NOT overwrite the original image)
        gray = cv2.cvtColor(self.image.image, cv2.COLOR_RGB2GRAY)
        
        # Extract the PCA features from the test image
        test_transformed = feature_extraction.transform_test_image(gray, self.mean, self.eigenvectors)
        
        # Ensure that test_transformed is a 2D array (1 sample, n_features)
        if len(test_transformed.shape) == 1:
            test_transformed = test_transformed.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Calculate distances between the test sample and the training samples
        distances = euclidean_distances(test_transformed, self.X_transformed)
        closest_idx = np.argmin(distances)
        
        print(f"Closest index: {closest_idx}, Label: {self.y[closest_idx]}, Distance: {distances[0][closest_idx]}")
        self.recognized_face_label.setText(f"{os.path.splitext(self.y[closest_idx])[0]}")
              
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())