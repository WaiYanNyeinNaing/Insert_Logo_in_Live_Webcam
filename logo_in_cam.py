import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
# This is preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Resize with Scale
logo = cv2.imread("image/TF.jpg")
scale = 10
width = logo.shape[0]//scale
height = logo.shape[1]//scale
logo = cv2.resize(logo, (height, width))

#Preprocessing + Mask
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
print(logo.shape)
print(mask.shape)

 
# The main loop
while True:
    _, frame = cap.read()
    # This is the processing
    #frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    #Insert Log       y:y+h  ,  x:x+w
    roi = frame[-width-10:-10, -height-10:-10]
    print(f"\r {roi.shape}",end="")
    roi[np.where(mask)] = 0
    roi += logo

    # Here we show the image in a window
    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask)
 
    # Check if q was pressed
    if cv2.waitKey(1) == ord('q'):
        break