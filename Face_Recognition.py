#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Face Detection and Recognition
# Haar Cascade and Eigenface
# Author: Nicko.creator0
# November 2022


# In[2]:


# Import required packages
import numpy as np
import cv2
from numpy import asarray
from matplotlib import pyplot as plt

# Load the xml file of Haar features
# All Haar Cascade files in my github repository:
# https://github.com/nickocreator0/haar-cascade-files
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

testing_IDs = []
training_IDs = []
testing_faces = []
training_faces = []


# In[3]:


# Contrast Stretching Function
# Preprocessing - before recognition
# The formula:
# Pnew = [(Pin - Pmin)/(Pmax-Pmin)]*255

def contrast_stretch(img, scale):
    #Create a new image to store the stretch pixels in
    stretched = img.copy()#np.zeros_like(img.shape, img.dtype)
    Pmin = np.min(img)
    Pmax = np.max(img)

    for i in range(img.shape[0]): # height
        for j in range(img.shape[1]): # width
            Pnew = (img[i][j] - Pmin)/(Pmax - Pmin)*scale#(max_new - min_new) + min_new
            if Pnew < 0:
                Pnew = 0
            elif Pnew > 255:
                Pnew = 255
            stretched[i][j] = Pnew
    return stretched


# In[4]:


# Detect-face Function

def face_detect_add(f_name, flag):
    detected = False #nothing's detected yet
    color_img = cv2.imread(f_name)

    converted = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    ### Contrast stretch for detection; uncomment to see the result of contrast on detection
    #contrasted = contrast_stretch(converted, 255)
    ###
    
    ### Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    gray_img = cv2.filter2D(src = converted, ddepth = -1, kernel = kernel)
    ### 
    
    face_detected = face_detector.detectMultiScale(gray_img, 1.2, 6, minSize = (50, 50))

    if len(face_detected) == 1: #if a face's detected
        
        # Set the Region Of Interest
        for (x, y, w, h) in face_detected:
            roi_gray = gray_img[y: y+h, x: x+w]
        roi_gray = cv2.resize(roi_gray, dsize = (100, 100), interpolation = cv2.INTER_CUBIC)
        
        if flag == False: #If training's being done
            
            # Add to training_faces list 
            training_faces.append(roi_gray)
        
        else: # If testing's being done
            # Add to testing_faces list 
            testing_faces.append(roi_gray)
        
        # Flip this flag if a face's been detected
        detected = True

    return detected


# In[5]:


import glob
from natsort import natsorted

# Load images to train the recognizers
directory = ['Faces/01/', 'Faces/02/', 'Faces/03/', 'Faces/04/', 'Faces/05/']

is_detected = False

# Get the file name and pass it to face_detect_add function
for index, item in enumerate(directory[:]):
    for file_name in natsorted(glob.iglob(item + '/**/*.jpg', recursive = True)):
        is_detected = face_detect_add(file_name, False)
        
        # If the face_detect_add detects a face in this image
        if is_detected == True:
            
            #Add to training IDs
            training_IDs.append(index + 1)

del is_detected
del file_name


# In[6]:


#Load images to test the recognizers
is_detected = False

for index, file_name in enumerate(natsorted(glob.iglob('Faces/Test/**/*.jpg', recursive = True))):
    is_detected = face_detect_add(file_name, True)
    
    #If the face_detect_add detects a face in this image
    if is_detected == True:
        
        #Add to testing IDs
        testing_IDs.append(int((index/10)) + 1)
        
del is_detected
del file_name


# In[7]:


# Detection results

# The percentage of faces in test set having been detected
hit_rate = 100*len(testing_IDs)/50

print(f"Training samples: {len(training_IDs)}")
print(f"Testing samples: {len(testing_IDs)}")
print(f"Hit Rate: {hit_rate}%")

del hit_rate


# In[9]:


# RECOGNITION

from sklearn.decomposition import PCA
import math

# Initialize face recognizers

# Eigenface
e_recognizer = cv2.face.EigenFaceRecognizer.create(80, math.inf)

# FisherFace
#f_recognizer = cv2.face.FisherFaceRecognizer.create(0, 3500)

# LBPH
#l_recognizer = cv2.face.LBPHFaceRecognizer.create(1, 8, 8, 8, 100)


# In[ ]:


# Train the recognizers
import time

# OpenCV requires the the labels to be numpy arrays
faces_to_train = []

for sample in training_faces:
    
    # Contrast Stretching
    contrasted = contrast_stretch(sample, 60)
    temp = cv2.GaussianBlur(contrasted, (5,5), 0)
    ###
    faces_to_train.append(temp)
    
del temp
del sample

# Set the starting time
t0 = time.time()

# Train the EigenFace recognizer
e_recognizer.train(faces_to_train, np.array(training_IDs))

# Calculate running time for Recognition
t_train = time.time() - t0

#f_recognizer.train(faces_to_train, np.array(training_IDs))
#l_recognizer.train(faces_to_train, np.array(training_IDs))
del contrasted


# In[ ]:


# To check if prediction works the way it should
#ID, res_temp = e_recognizer.predict(faces_to_train[2])
#print(f"ID = {ID}")
#print(f"res_temp = {res_temp}")


# In[ ]:


# Prediction/Testing

e_dist = 0
recognized_correctly = 0
accuracy = 0

faces_to_test = []

for sample in testing_faces:
    
    #Contrast Stretching
    contrasted = contrast_stretch(sample, 60)
    ###
    temp = cv2.GaussianBlur(contrasted, (5,5), 0)
    faces_to_test.append(temp)
    
    # Set the starting time
    t0_ = time.time()
    
for index, test_case in enumerate(faces_to_test):
    # Get the ID and the distance for each image in test set
    ID, e_result = e_recognizer.predict(test_case)
    if ID == 1:
        plt.title(f"Arnold Schwarzenegger/dist = {e_result}")
        plt.imshow(test_case)
        # cv2.waitKey(0); could be displayed with openCV also 
        plt.show()
    
    elif ID == 2:
        plt.title(f"Dwayne Johnson/dist = {e_result}")
        plt.imshow(test_case)
        #cv2.waitKey(0)
        plt.show()
    
    elif ID == 3:
        plt.title(f"Jason Statham/dist = {e_result}")
        plt.imshow(test_case)
        #cv2.waitKey(0)
        plt.show()
    
    elif ID == 4:
        plt.title(f"Nicolas Cage/dist = {e_result}")
        plt.imshow(test_case)
        #cv2.waitKey(0)
        plt.show()
    
    elif ID == 5:
        plt.title(f"Sylvester Stallone/dist = {e_result}")
        plt.imshow(test_case)
        #cv2.waitKey(0)
        plt.show()

    # Count the number of faces recognized correctly
    if ID == testing_IDs[index]:
        recognized_correctly += 1
    
    e_dist += e_result
    #l_dist += l_result
    
# Calculate the accuracy: (Original num of faces - faces recognized successfully)/Original num of faces
accuracy = (len(testing_faces)-recognized_correctly)/len(testing_faces)

# Calculate running time for Recognition
t_test = time.time() - t0_
e_dist /= len(faces_to_test)

print(f"Training time = {t_train}")
print(f"Recognition Time = {t_test}")
print(f"Eigenface total distance= {e_dist}")
print(f"Number of faces recognized successfully= {recognized_correctly}")
print(f"Accuracy= {accuracy}")


# In[ ]:


import math
def euclidean_distance (point1, point2):
        return math.sqrt(((point2[0] - point1[0]) * (point2[0] - point1[0])) + ((point2[1] - point1[1]) * (point2[1] - point1[1])))


# In[ ]:


import math
from scipy.ndimage.interpolation import rotate

def face_alignment(gray_image):
    eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
    eyes = eye_detector.detectMultiScale(gray_image)
    rotated = gray_image.copy()
    
    if len(eyes) >= 2:
        eye1 = eyes[0]
        eye2 = eyes[1]
        left_eye = eye1
        right_eye = eye2
        for index, (eye_x, eye_y, eye_w, eye_h) in enumerate(eyes):
            if index == 0:
                eye1 = (eye_x, eye_y, eye_w, eye_h)
            elif index == 1:
                eye2 = (eye_x, eye_y, eye_w, eye_h)
        #determining the left and right eyes
        if eye1[0] > eye2[0]:
            left_eye = eye2
            right_eye = eye1
        #finding the coordinates of the centroids
        #left
        left_eye_cent = (int((left_eye[0] + left_eye[2])/2), int((left_eye[1] + left_eye[3])/2))
        left_eye_x = left_eye_cent[0]
        left_eye_y = left_eye_cent[1]

        #right
        right_eye_cent = (int((right_eye[0] + right_eye[2])/2), int((right_eye[1] + right_eye[3])/2))
        right_eye_x = right_eye_cent[0]
        right_eye_y = right_eye_cent[1]

        #The rotation
        #if the y-coordinate of the left eye's greater that y-coordinate of the right one, 
        #the rotation needs to be in clock-wise direction and vice versa
        #We suppose a triangle formed by left and right eyes and a 3rd point
        #then the angle of rotation will be found
        if left_eye_y > right_eye_y:
            tri_point = (right_eye_x, left_eye_y) #the line between forms the triangle
            direction = -1
        else:   #rotate counterclock-wise
            tri_point = (left_eye_x, right_eye_y) 
            direction = 1
        #using Euclidean distance, we can find the angle between the two lines 
        #and then rotate, thus align the eyes
        #  cos(x) = (b^2+ c^2– a^2 ) / (2bc)
        #a, b and c are the distances between eyes centers and the 3rd point
        a = euclidean_distance(left_eye_cent, tri_point) 
        b = euclidean_distance(right_eye_cent, tri_point)
        c = euclidean_distance(right_eye_cent, left_eye_cent)

        cos_of_angle = (b*b + c*c - a*a)/(2*b*c)
        rotation_angle = (np.arccos(cos_of_angle) * 180) / math.pi

        if direction == -1:
            rotation_angle = 90 - rotation_angle
        else:
            rotation_angle = -(90 - rotation_angle)
        #Rotate the image
        rotated = rotate(gray_image, angle = (direction * rotation_angle))
    return rotated

