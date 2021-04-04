import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

input_path = 'image/iron.jpeg'
sav_path = 'result'
scale = 1

#get filename
full_name = input_path.split("/")[1]
name = full_name.split('.')[0]

frame = cv2.imread(input_path)
img = cv2.resize(frame,(frame.shape[1]//scale,frame.shape[0]//scale))  #resize image by half   
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#select ROI function
roi = cv2.selectROI(img)
#print rectangle points of selected roi
print(roi)

#assign
rect = roi

#GrabCut Segmentation
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv2.imshow("frame",img)
cv2.imwrite(f"{sav_path}/segment_{name}_result.png",img)
cv2.imwrite(f"{sav_path}/mask_{name}_result.png",mask2*255)
cv2.waitKey(0)





