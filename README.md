# Face-Detection-and-Recognition

This project implements a basic face recognition system using Principal Component Analysis (PCA). It detects and recognizes faces from grayscale images by projecting them into a lower-dimensional eigenspace and comparing the test image with stored training faces.

## Features
- Load and preprocess face images (resizing and grayscale conversion)
- PCA-based dimensionality reduction
- Faces Detection
- Face recognition via Euclidean distance comparison in PCA space

## Faces Detection
![image](https://github.com/user-attachments/assets/f0500eb2-17a7-4486-8131-374b01dcb4c3)

## Face Recognition
![image](https://github.com/user-attachments/assets/9e9bf203-83e4-490f-8328-4db6085e6486)

## Techniques Used
- Principal Component Analysis (PCA)
- Euclidean distance matching
- Image preprocessing with OpenCV

## Project Structure
- `feature_extraction.py`: Handles PCA training and test image transformation
- `program.py`: Main interface for training and recognizing faces
- `data/`: Folder containing training face images

## Requirements
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy  

## How to Run
1. Add training images to the `data/` folder.
2. Run the main script:
   ```bash
   python program.py
