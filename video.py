import cv2
import numpy as np

cap = cv2.VideoCapture(0)

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces_detections = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)
    
    for (x,y,w,h) in faces_detections:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=5)

    cv2.imshow('frame', frame)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
      
      
cap.release()     
cv2.destroyAllWindows()
