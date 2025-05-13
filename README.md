# Face-Detection-and-Recognition

This project implements a basic face recognition system using Principal Component Analysis (PCA). It detects and recognizes faces from grayscale images by projecting them into a lower-dimensional eigenspace and comparing the test image with stored training faces.

## Features
- Load and preprocess face images (resizing and grayscale conversion)
- PCA-based dimensionality reduction
- Faces Detection
- Face recognition via Euclidean distance comparison in PCA space

## Techniques Used
- Principal Component Analysis (PCA)
- Euclidean distance matching
- Image preprocessing with OpenCV

## Faces Detection
![Screenshot 2025-05-10 234858](https://github.com/user-attachments/assets/6fd42252-7212-4835-90f2-9a063f47734d)

## Face Recognition
![Screenshot 2025-05-10 234837](https://github.com/user-attachments/assets/735410b5-9691-4bb0-a506-2b779019c1e7)

## Confusion Matrix:
![Screenshot 2025-05-10 234911](https://github.com/user-attachments/assets/bab9f560-0608-4fa7-81ed-77553a066247)

## ROC Curve: 
![Screenshot 2025-05-10 234924](https://github.com/user-attachments/assets/4127d65e-f8cc-4405-b90e-fc94716f47f1)

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
