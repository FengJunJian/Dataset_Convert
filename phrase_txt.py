import numpy
import cv2
AnnoPath='H:\\c_dataset\\rssrai2019_object_detection\\val\\labelTxt\\P0003.txt'
ImgPath='H:\\c_dataset\\rssrai2019_object_detection\\val\\images\\P0003.png'
Rects=[]
Classes=[]
with open(AnnoPath,'r') as f:
    while True:
        line=f.readline()
        if line is '':
            break
        else:
            s=line.split()
            if(len(s)<10):
                continue
            Rect=[(int(float(s[0])),int(float(s[1]))),(int(float(s[2])),int(float(s[3]))),(int(float(s[4])),int(float(s[5]))),(int(float(s[6])),int(float(s[7])))]
            category=s[8]
            difficult=int(s[9])
            Rects.append(Rect)
            Classes.append(category)
            #print(line.strip())

img=cv2.imread(ImgPath)
for i in range(len(Rects)):
    cv2.line(img,Rects[i][0],Rects[i][1],(0,0,255),2)
    cv2.line(img, Rects[i][1], Rects[i][2], (0, 0, 255), 2)
    cv2.line(img, Rects[i][2], Rects[i][3], (0, 0, 255), 2)
    cv2.line(img, Rects[i][3], Rects[i][0], (0, 0, 255), 2)
    cv2.putText(img,Classes[i],Rects[i][0],cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
    #img=cv2.rectangle(img,r[0],r[2],(0,0,255),2)
cv2.imshow('a',img)
cv2.waitKey()