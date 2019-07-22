#! /usr/bin/python
# -*- coding:UTF-8 -*-
import os,sys
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS=None

# 文件路径
src_img_dir = "H:\\c_dataset\\rssrai2019_object_detection\\val\\images\\"
src_txt_dir = "H:\\c_dataset\\rssrai2019_object_detection\\val\\labelTxt\\"
src_xml_dir = "H:\\c_dataset\\rssrai2019_object_detection\\val\\labelXml\\"

if not os.path.exists(src_xml_dir):
    os.makedirs(src_xml_dir)
img_Lists = glob.glob(src_img_dir + '*.png')

img_basenames = []  # e.g. 100.jpg
for item in img_Lists:
    img_basenames.append(os.path.basename(item))

img_names = []  # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)

for img in img_names:
    img_file=src_img_dir + img + '.png'
    try:
        im = Image.open(img_file)
    except Exception:
        print(img_file)
        continue
    width, height = im.size
    # open the crospronding txt file
    gt = open(src_txt_dir  + img + '.txt').read().splitlines()
    # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    # os.mknod(src_xml_dir + '/' + img + '.xml')
    xml_file = open((src_xml_dir + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>remote sensing</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.png' + '</filename>\n')
    xml_file.write('    <path>'+img_file+'</path>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>Unknown</database>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('    <segmented>'+'0'+'</segmented>\n')

    # write the region of image on xml file
    for img_each_label in gt:
        spt = img_each_label.split()  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        if (len(spt) < 10):
            continue
        Rect = [(int(float(spt[0])), int(float(spt[1]))), (int(float(spt[2])), int(float(spt[3]))),
                (int(float(spt[4])), int(float(spt[5]))), (int(float(spt[6])), int(float(spt[7])))]
        category = spt[8]
        difficult = int(spt[9])

        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(category) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(Rect[0][0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(Rect[0][1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(Rect[2][0]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(Rect[2][1]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')