from PyQt5 import QtWidgets, QtCore, uic
import sys
from PyQt5.QtGui import *
import numpy as np
import cv2
from sklearn.metrics.pairwise import euclidean_distances
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


from Image import Image
import feature_extraction

test_data_folder = "test_data/"
train_data_folder = "train_data/"
# image_size = (50,50)
# num_components = 20
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('gui.ui', self)
            
        # buttons connection
        self.upload_button.clicked.connect(self.upload_image)
        self.face_detection_button.clicked.connect(self.detect_faces)
        self.face_recognitio_button.clicked.connect(lambda: self.recognize_face(self.image.image, key=1))
        self.confusion_matrix_button.clicked.connect(self.draw_confusion_matrix)
        
        # load & Train faces from data set
        X_train, self.labels = feature_extraction.load_faces(train_data_folder)
        self.eigenvectors, self.eigenvalues, self.X_transformed, self.mean = feature_extraction.train_faces(X_train)
        print("Training DONEEEE")
        
        # load test data
        self.X_test, self.y_test = feature_extraction.load_faces(test_data_folder)
        
        
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
        if len(self.image.image.shape) == 3 and self.image.image.shape[2] == 3:
            gray = cv2.cvtColor(self.image.image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.image.image  # Already grayscale

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
            
        
    def recognize_face(self, image, key=0):
        # Convert to grayscale (do NOT overwrite the original image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image  # Already grayscale
        
        # Extract the PCA features from the test image
        test_transformed = feature_extraction.transform_test_image(gray, self.mean, self.eigenvectors)
        
        # Ensure that test_transformed is a 2D array (1 sample, n_features)
        if len(test_transformed.shape) == 1:
            test_transformed = test_transformed.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Calculate distances between the test sample and the training samples
        distances = euclidean_distances(test_transformed, self.X_transformed)
        closest_idx = np.argmin(distances)
        
        if key == 1:
            print(f"Closest index: {closest_idx}, Label: {self.labels[closest_idx]}, Distance: {distances[0][closest_idx]}")
            self.recognized_face_label.setText(f"{os.path.splitext(self.labels[closest_idx])[0]}")
        else:
            return self.labels[closest_idx]    

            
    def draw_confusion_matrix(self):
        y_pred = []
        for file_name in os.listdir(test_data_folder):
            file_path = os.path.join(test_data_folder, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None: 
                continue

            pred = self.recognize_face(image)
            y_pred.append(pred)
        
        print(f"y_test:  {self.y_test}")
        print(f"y_pred:  {y_pred}")
        # Get all unique sorted labels (from both test and predictions)
        all_labels = sorted(list(set(self.y_test + y_pred)))
        
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=all_labels)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        
        # Set labels
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_yticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_yticklabels(all_labels)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add text annotations
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                ax.text(j, i, cm[i, j], 
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
        
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()                
                      
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())