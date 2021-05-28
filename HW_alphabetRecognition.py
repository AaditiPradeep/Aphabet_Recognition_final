import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

#Fetching the data

#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R','S', 'T','U','V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    # try and catch: eg when internet is not working it shows message(exceptions) if not it will execute next step
    ret, frame = cap.read()
    # ret: stores boolean values true(when camera is on) false(when camera is off)
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    #upper_left, bottom_right : four sides of rectangle
    
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    #print(roi)

    #Converting cv2 image to pil format
    im_pil = Image.fromarray(roi) # converts the roi with image processiing capabilities

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L') # L- format : each pixel in the roi will represent a single random value from 0 to 255
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)# ANTIALIAS: no distortion of image and image will be clear

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) #https://www.geeksforgeeks.org/numpy-percentile-in-python/ 
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)#https://www.geeksforgeeks.org/numpy-clip-in-python/
    max_pixel = np.max(image_bw_resized_inverted)#https://www.geeksforgeeks.org/numpy-maximum-in-python/
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel # for scaling
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)# reshaping the image to 1,784, storing the number in test_sample
    test_pred = clf.predict(test_sample)#predictng the values
    print("Predicted class is: ", test_pred)#printing it

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'): # if camera is closed, then stop running the function
      break
  except Exception as e:
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






        
        

        

        
        

        



