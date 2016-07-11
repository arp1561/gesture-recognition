import cv2
import numpy as np
import pygame
from pygame.locals import *

#init video capture
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

#init the value for one two three and four 
one=False
two=False
three=False
four=False
onlyOne=0

#running the main loop for getting the frames and then changing the counters as well as stacking 
while storing_data:
      ret,img=cap.read()
      h,w,c=img.shape
      if onlyOne==0:
          print h," ",w
          onlyOne+=1
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      unroll  = gray.reshape(1,76800).astype(np.float32)
      cv2.imshow('frame',img)
      if cv2.waitKey(1) & 0xFF==ord('q'):
            break



      for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
	            key_input = pygame.key.get_pressed()
                    if key_input[pygame.K_1]:
				one,two,three,four = True,False,False,False
				print "One"
				cv2.imshow('frame',img)
				cv2.imwrite('training_images/one'+str(count)+'.jpg',img)
				count+=1
		    elif key_input[pygame.K_2]:
				one,two,three,four = False,True,False,False
				print "Two"
				cv2.imwrite('training_images/two'+str(count)+'.jpg',img)
				count+=1
		    elif key_input[pygame.K_3]:
				one,two,three,four = False,False,True,False
				print "Three"
				cv2.imwrite('training_images/three'+str(count)+'.jpg',img)
				count+=1
		    elif key_input[pygame.K_4]:
				one,two,three,four = False,False,False,True
				print "Four"
				cv2.imwrite('training_images/three'+str(count)+'.jpg',img)
				count+=1
		    elif key_input[pygame.K_x]:
				print "All data collected"
				storing_data = False
				pygame.display.quit()
				break

		if(one==True):
			print "Stacking for one"
			training_image_array=np.vstack((training_image_array,unroll))
			output_array=np.vstack((output_array,true_label[0]))
			frame_count+=1
		if(two==True):
			print "Stacking for two"
			training_image_array=np.vstack((training_image_array,unroll))
			output_array=np.vstack((output_array,true_label[1]))
			frame_count+=1
		if(three==True):
			print "Training for three"
			training_image_array=np.vstack((training_image_array,unroll))
			output_array=np.vstack((output_array,true_label[2]))
			frame_count+=1
		if(four==True):
			print "Training for four"
			training_image_array=np.vstack((training_image_array,unroll))
			output_array=np.vstack((output_array,true_label[3]))
                    	frame_count+=1
cv2.destroyAllWindows()
training_image_array=training_image_array[1:,:]
output_array=output_array[1:,:]

print training_image_array.shape
print output_array.shape
#print output_array

np.savez('training.npz', training_image_array = training_image_array, output_array = output_array)
	

