import cv2
import numpy as np

cap=cv2.VideoCapture(0)

cascade=cv2.CascadeClassifier("F:\\programs\\python\\opencv\\haarcascades\\haarcascade_frontalface_default.xml")

while True:
    _, img=cap.read()
    img1=cv2.flip(img,1)
    gray_image=cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
    face=cascade.detectMultiScale(gray_image,1.1,10)
    for (x,y,w,h) in face:
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("image",img1)
    key=cv2.waitKey(1)
    if key==ord("s"):
        break
    
cv2.destroyAllWindows()