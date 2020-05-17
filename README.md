# CellTrack
This project is the implement of the algorithm proposed in manuscript "An Advanced Framework for Time-lapse Microscopy Image Analysis". It contains two tasks to process time-lapse microscopy images. The first part is to classify cancer cells into live and dead categories. The second part is to detect phagocytoses in these images.

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
The files in folder cell_classification are the codes to classify cancer cells into live and dead categories.

### **cell_detect.py**
This file implements how to detect cells in the images. It contains several steps, converting color images to gray images, converting gray images to binary images, finding contours in binary images, determining which kinds of objects are cells, calculating the shape of cells.
### **cell_classify.py**
This file implements how to classify cells in the images. It contains several steps, tracking cells through contiuous images, determining which are live or dead cells seperatly.
### **main.py**
This is the entry point of the program. The input data are time-lapse microscopy images. The output data is a video in which the cells are marked live or dead.

### How to Run
Users need to run the code in Ubuntu Envirenment. After prepare the input data, just execute the following command:
$./main.py

## Phagocytosis Detection
The files in folder phagocytosis_detection are the codes to detect phagocytosis in the images. 

### **cell_detect.py**
This file implements how to detect cells in the images. It contains several steps, converting color images to gray images, converting gray images to binary images, finding contours in binary images, determining which kinds of objects are cells, calculating the shape of cells.
### **phagocytosis_detect.py**
This file implements how to detect phagocytosis in the continuous images. It contains the applies of DBSCAN, linear regression and how to determine if a cluster contains phagocytosis.

### **main.py**
This is the entry point of the program. The input data is one video made up of time-lapse microscopy images. The output data is a video in which cells are clustered and clusters are marked if it contains phagocytosis.

### How to Run
Users need to run the code in Ubuntu Envirenment. After prepare the input data, just execute the following command:
$./main.py


