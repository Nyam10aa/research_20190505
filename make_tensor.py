#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from PIL import Image
import numpy as np
import sys

def format_image(image):
	#image = np.array(Image.open(image, 'r').resize((18, 180)))
	image = np.array(Image.open(image, 'r'))
	return image

def label_settings(x):
    d = np.zeros(90)
    for i in range(len(x)):
    	if x[i]=='1':
    		d[i*5+int(x[i])-1]=1.0
    	if x[i]=='2':
    		d[i*5+int(x[i])-1]=1.0
    	if x[i]=='3':
    		d[i*5+int(x[i])-1]=1.0
    	if x[i]=='4':
    		d[i*5+int(x[i])-1]=1.0
    	if x[i]=='5':
    		d[i*5+int(x[i])-1]=1.0
    return d/18

def label_settings_3class(x):
        d = np.zeros(18*3)
        for i in range(len(x)):
                if x[i]=='1':
                        d[i*3+int(x[i])-1]=1.0
                if x[i]=='2':
                        d[i*3+int(x[i])-1]=1.0
                if x[i]=='3':
                        d[i*3+int(x[i])-2]=1.0
                if x[i]=='4':
                        d[i*3+int(x[i])-3]=1.0
                if x[i]=='5':
                        d[i*3+int(x[i])-3]=1.0
        print(d, len(d))
        return d/18

pic_name = pd.read_csv("pic_name.csv")["filename"].tolist()

def random_choice():
	pic_name1=[]
	for p in pic_name:
		pic_name1.append(p)
	print(pic_name1)
	print(len(pic_name))
	length=len(pic_name1)
	te=np.random.choice(length, int(length/10*3), replace=False).tolist()
	return te

test_name=[]
learn_name=[]


start=121
end=140
start = int(sys.argv[1])
end = int(sys.argv[2])
print(start,end)
def specific_choice(start, end):
	te=[]
	for i in range(start,end+1):
		te.append("%03d"%i)	
	for pic in pic_name:
		if pic[0:3] in te:
			test_name.append(pic)
		else:
			learn_name.append(pic)

specific_choice(start,end)
print(len(test_name))
print(len(learn_name))

learn_labels = []
learn_images = []
test_labels = []
test_images = []

for t in test_name:
	test_images.append(format_image("pic/"+t))
	test_labels.append(label_settings_3class(t.split("_")[1][0:18]))

for l in learn_name:
	learn_images.append(format_image("pic/"+l))
	learn_labels.append(label_settings_3class(l.split("_")[1][0:18]))

print(len(learn_images))
print(len(learn_images[0]))
print(len(learn_images[0][0]))
print(len(learn_images[0][0][0]))

print("Total learn_dataset: " + str(len(learn_images)))
print("Total test_dataset: " + str(len(test_images)))

np.save("%03d_%03d_for_test/learn_data.npy" % (start,end), learn_images)
np.save("%03d_%03d_for_test/learn_labels.npy" % (start,end), learn_labels)
np.save("%03d_%03d_for_test/test_data.npy" % (start,end), test_images)
np.save("%03d_%03d_for_test/test_labels.npy" % (start,end), test_labels)


#print len(pic_name)

