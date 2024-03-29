# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:46:53 2021

@author: Souradeep
"""

# Importing necessary libraries
import keras
import cv2
import numpy as np

# Loading pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Setting up video stream from webcam
url = 'http://192.168.0.143:8080/video'
vid = cv2.VideoCapture(0)

# Counter to track frames processed
c = 0

# Loop to continuously process video frames
while(vid.isOpened()): 
      
    # Capture the video frame by frame
    name, im = vid.read()
    
    # Display the captured frame
    cv2.imshow("Antispoof", im)
    
    # Increment frame counter
    c = c + 1
    
    # Process frame if successfully captured
    if name:
        
        # Process every 20th frame
        if (c % 20 == 0):
            
            # Flip the image horizontally for better viewing
            image = cv2.flip(im, 1)
            
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.3,
                                                   minNeighbors=3,
                                                   minSize=(96, 96))
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
                
            # Resize image to match input size of the model
            image = cv2.resize(image, (96, 96))
            
            # Reshape image to match input shape expected by the model
            image = image.reshape(1, 96, 96, 3)
            
            # Load the pre-trained model
            model = keras.models.load_model('antispoofmobilenet2.h5')
            
            # Make predictions using the model
            pred = model.predict(image[:][:])
            
            # Convert prediction to string for comparison
            p = str(np.round(pred[0][1]))
            
            # Print the prediction
            print(p)
            
            # Display prediction on the frame
            if p == "0.0":
                x = cv2.putText(im, "Fake", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                x = cv2.putText(im, "Real", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                
            # Display the annotated frame
            cv2.imshow("Antispoof", x)
            
            # Break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

# Release video capture object
vid.release() 

# Close all windows
cv2.destroyAllWindows()
        
        
               
