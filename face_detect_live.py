import cv2
import numpy as np

#classifier
detector= cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
detector1= cv2.CascadeClassifier('xml/haarcascade_eye.xml')
detector2= cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')

 
cap = cv2.VideoCapture(0)
# This is preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  
# The main loop
while True:
    _, frame = cap.read()
    # This is the processing
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    #Face_Detection
    gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray_face, 1.3, 5)

    #Draw Rect on Frame
    for (x,y,w,h) in faces:                             #for x,y,width,height in faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  #(R,G,B color) #2 = thickness of the rectangle

        #Crop Face ROI
        face_roi = frame[y:y+h,x:x+w]

    #Insert Live Logo
    size = 200
    face_roi = cv2.resize(face_roi, (size, size))
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    roi = frame[-size-10:-10, -size-10:-10]
    roi[np.where(mask)] = 0
    roi += face_roi

    # Here we show the image in a window
    cv2.imshow("Webcam", frame)
 
    # Check if q was pressed
    if cv2.waitKey(1) == ord('q'):
        break