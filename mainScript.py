import cv2
import numpy as np
import pygame
from pygame.locals import *


cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

count = 0 
frame_count=0

storing_data=True

#defing arrays for training image and output gestures
training_image_array=np.zeros((1,76800))
output_array = np.zeros((1,4),float)

#creating an array for inserting into output array
true_label=np.zeros((4,4),float)
for i in range(4):
	true_label[i,i]=1
#init pygame
pygame.init()
display = pygame.display.set_mode((300,200))
display.fill((0,0,0))

#creating network
layer_size=np.int32([76800,64,32,4])
neural = cv2.ANN_MLP()
neural.create(layer_size)
neural.load('mlp_xml/mlp.xml')


#init the value for one two three and four 
one=False
two=False
three=False
four=False
onlyOne=0

while storing_data:
    ret,img = cap.read()
    h,w,c = img.shape
    if onlyOne==0:
        print h," ",w
        onlyOne+=1
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    unroll  = gray.reshape(1,76800).astype(np.float32)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    ret,resp=neural.predict(unroll)
    prediction=resp.argmax(-1)
#    print "Predicted Value = ",prediction
    if prediction == 0:
        print "One"
    elif prediction == 1:
        print "Two"
    elif prediction == 2:
        print "Three"
    elif prediction == 3:
        print "Four"

cv2.destroyAllWindows()
cap.release()

