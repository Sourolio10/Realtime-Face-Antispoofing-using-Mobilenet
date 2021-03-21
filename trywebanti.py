# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:46:53 2021

@author: USER
"""

import keras
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
url = 'http://192.168.0.143:8080/video'
vid = cv2.VideoCapture(0)
c=0
while(vid.isOpened()): 
      
    # Capture the video frame 
    # by frame 
    name,im = vid.read()
    cv2.imshow("Antispoof",im)
    c=c+1
    if name:
        if(c%20==0):
            image=cv2.flip(im,1)
            gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(96, 96)
        ) 
            for(x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),1)
            #image=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image,(96, 96))
            image=image.reshape(1,96,96,3)
            model = keras.models.load_model('antispoofmobilenet2.h5')
            pred = model.predict(image[:][:])
            p=str(np.round(pred[0][1]))
            print(p)
            
            if p== "0.0":
                x=cv2.putText(im,"Fake" , (50,150), cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,255), 2, cv2.LINE_AA)
            else:
                x=cv2.putText(im,"Real" , (50,150), cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("Antispoof",x)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        #cv2.imshow("Antispoof",image)
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
        
        
               