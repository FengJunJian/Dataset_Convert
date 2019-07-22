import json
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import cv2
import numpy as np
import glob

json_data={}
src_img_dir = "H:\\c_dataset\\rssrai2019_object_detection\\val\\images\\"
src_txt_dir = "H:\\c_dataset\\rssrai2019_object_detection\\val\\labelTxt\\"
img_Lists = glob.glob(src_img_dir + '*.png')
#image_path='H:/c_dataset/rssrai2019_object_detection/val/images/%s.png'
#txt_path='H:/c_dataset/rssrai2019_object_detection/val/labelTxt/P0003.txt'
_,basename=os.path.split(src_txt_dir+'P0003.txt')
basename=os.path.splitext(basename)[0]

image=src_img_dir+basename+'.png'
file_size=os.path.getsize(image)

try:
    im = Image.open(image)
except Exception:
    print(image)
    os._exit(1)
    #continue
width, height = im.size
file_key = basename+'.png' + str(file_size)
file_value = {}

json_data[file_key] = file_value
file_value["fileref"] = ""
file_value["size"] = file_size
file_value["filename"] = basename + '.png'
file_value["base64_img_data"] = ""
file_value["file_attributes"] = {}

regions = {}
file_value["regions"] = regions

gt = open(src_txt_dir+'P0003.txt').read().splitlines()
categorys=[]#总类别
# write the region of image on xml file
i=0
for img_each_label in gt:
    spt = img_each_label.split()  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
    if (len(spt) < 10):
        continue
    x = []
    y = []
    x.append(int(float(spt[0])))
    x.append(int(float(spt[2])))
    x.append(int(float(spt[4])))
    x.append(int(float(spt[6])))

    y.append(int(float(spt[1])))
    y.append(int(float(spt[3])))
    y.append(int(float(spt[5])))
    y.append(int(float(spt[7])))

    #Rect = [(int(float(spt[0])), int(float(spt[1]))), (int(float(spt[2])), int(float(spt[3]))),
    #        (int(float(spt[4])), int(float(spt[5]))), (int(float(spt[6])), int(float(spt[7])))]
    category = spt[8]
    difficult = int(spt[9])
    region={}#一个目标
    shape={}
    regions[str(i)]=region
    region["shape_attributes"]=shape
    region["region_attributes"]={}
    shape["name"]="polygon"
    shape["all_points_x"]=x
    shape["all_points_y"]=y
    i=i+1

json.dump(json_data,open('demo.json', 'w'),indent=4,sort_keys=True)




