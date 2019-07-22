import json
import os
import cv2
import numpy as np
path='H:/c_dataset/rssrai2019_object_detection/val/images/P0003.png'
basename=os.path.basename(path)
file_size=os.path.getsize(path)
#path='H:/code/Mask_RCNN-master/balloon_dataset/balloon/train/34020010494_e5cb88e1c4_k.jpg'
_,filename=os.path.split(path)
json_path="demo.json"
#json_path="H:/code/Mask_RCNN-master/balloon_dataset/balloon/train/via_region_data.json"

with open(json_path,'r') as f:
    data=json.load(f)
    image = data[basename+str(file_size)]
    regions=image['regions']
    num=len(regions.keys())
    objects=[]
    for i in range(num):
        points=[]
        shape=regions[str(i)]
        shape=shape['shape_attributes']
        x_json=shape['all_points_x']
        y_json=shape['all_points_y']
        for j in range(len(x_json)):
            points.append([x_json[j],y_json[j]])
        objects.append(points)
    img=cv2.imread(path)
    img=cv2.drawContours(img,np.array(objects),-1,(255,255,255),-1)

cv2.imshow('a',img)
cv2.waitKey()
cv2.destroyAllWindows()