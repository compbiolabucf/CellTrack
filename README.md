# CellTrack
The algorithm proposed in manuscript "An Advanced Framework for Time-lapse Microscopy Image Analysis" can detect, track, and classify  cancer cells as well as detect phagocytosis in time-lapse Microscopy images. 

## Required Python packages
- Numpy
- Opencv
- sklearn
- statistics
- hungarian
- pandas
- matplotlib 
- multiprocessing

## Cell Classification
The files in folder cell_classification are the codes required to classify cancer cells into live and dead.

### **cell_detect.py**
This file implements detection of cells in the images. It contains several steps such as converting color images to gray scale images and gray scale images to binary images, finding contours in binary images, determining if a contour is actually a cell, and calculating the shape of the cells.
### **cell_classify.py**
This code classifies cells in the images. It contains several steps such as tracking cells through contiuous images, determining which cells are live and which are dead.
### **main.py**
This code takes time-lapse microscopy images as input data and give the user the classfication of cells as output. It calls cell_detect.py and cell_classify.py for the computation. 

### How to Run
Users need to run the code in Ubuntu Envirenment. After preparing the input data, execute the following command:

$./main.py

## Phagocytosis Detection
The files in folder phagocytosis_detection are the codes to detect phagocytosis in the images. 

### **cell_detect.py**
This file implements detection of cells in the images. It contains several steps such as converting color images to gray scale images and gray scale images to binary images, finding contours in binary images, determining if a contour is actually a cell, and calculating the shape of the cells.
### **phagocytosis_detect.py**
This code detects phagocytosis in continuous images. It contains the application of DBSCAN, linear regression and determination if a cluster contains phagocytosis.

### **main.py**
This code takes time-lapse microscopy images as input data and give the user a video in which cells are clustered and clusters are marked if it contains phagocytosis as output. It calls cell_detect.py and phagocytosis_detect.py for the computation. 

### How to Run
Users need to run the code in Ubuntu Envirenment. After preparing the input data, execute the following command:

$./main.py


