import numpy
import cv2
AnnoPath='H:\\c_dataset\\rssrai2019_object_detection\\val\\labelTxt\\P0003.txt'
ImgPath='H:\\c_dataset\\rssrai2019_object_detection\\val\\images\\P0003.png'

with open(AnnoPath,'r') as f:
    while True:
        line=f.readline()
        if line is '':
            break
        else:
            print(line.strip())