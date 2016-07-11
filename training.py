import cv2
import numpy as np
import glob

training_data = np.zeros((1,76800))
labels = np.zeros((1,4),'float')
train = glob.glob('training.npz')
print train
for i in train:
	with np.load(i) as data:
		print data.files
		training_temp = data['training_image_array']
		labels_temp = data['output_array']
	training_data = np.vstack((training_data,training_temp))
	labels = np.vstack((labels,labels_temp))

training_data = training_data[1:,:]
labels = labels[1:, :]

print training_data.shape
print labels.shape

e1 = cv2.getTickCount()

layer_size = np.int32([76800,64,32,4])

neural = cv2.ANN_MLP()
neural.create(layer_size)
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,600,0.0001)
params = dict(term_crit=criteria,train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,bp_dw_scale=0.001,bp_moment_scale=0.0)

print "training MLP...."

iterations = neural.train(training_data,labels,None,params=params)

e2 = cv2.getTickCount()

time_taken = (e2-e1)/cv2.getTickFrequency()

print "Time taken to train ",time_taken

#saving params
neural.save('mlp_xml/mlp.xml')

print "Total number of iteration",iterations

ret,resp = neural.predict(training_data)
predict=resp.argmax(-1)
print "prediction :",predict
true_labels = labels.argmax(-1)

print "True Labels :",true_labels

print "Testing....."

train_rate = np.mean(predict==true_labels)

print "Training Rate :",(train_rate*100)


