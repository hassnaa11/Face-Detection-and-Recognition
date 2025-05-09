import numpy as np
import os
import cv2

def load_faces(folder="data/", image_size=(30,30)):
    X = []
    y = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None: 
            continue
        
        image = cv2.resize(image, image_size)
        X.append(image.flatten())
        y.append(file_name)
    
    return X, y    

 
def train_faces(X_train):
    X_train = np.array(X_train)  # Shape: (num_samples, num_features)
    sorted_eigenvectors, sorted_eigenvalue, X_transformed,mean = pca(X_train)
    return sorted_eigenvectors, sorted_eigenvalue, X_transformed, mean   
        

def pca(X, num_components=3):
    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # calculate Covariance Matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # calculate eigen values and vectors
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)
    
    # sort eigen values and vectors
    sorted_indecies = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_indecies]
    sorted_eigenvectors = eigen_vectors[:, sorted_indecies]

    # Transform the data
    selected_vectors = sorted_eigenvectors[:, :num_components]
    X_transformed = X_centered @ selected_vectors

    return sorted_eigenvectors, sorted_eigenvalue, X_transformed,mean


def transform_test_image(test_image, mean, eigenvectors, image_size=(30, 30), num_components=3):
    test_image_resized = cv2.resize(test_image, image_size) # resize image
    test_image_flattened = test_image_resized.flatten() # flatten image

    # Center the test image by subtracting the mean
    test_image_centered = test_image_flattened - mean

    # Project the centered test image onto the eigenvectors
    test_transformed = test_image_centered @ eigenvectors[:, :num_components]

    return test_transformed
