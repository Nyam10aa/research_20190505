#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from PIL import Image
import boto3
import io
import numpy as np
import random
import sys

err_case=["085009","085012","085013"]

path = 'divided_by_20_additional/'

start = int(sys.argv[1])-1
end = int(sys.argv[2])-1

name = []
name1 = []

for i in range(start, end+1):
        for j in range(20):
                print(i)
                if not ("%03d%03d" % (i+1,j+1)) in err_case:
                	key = ("result_step_%03d%03d.csv" % (i+1,j+1))
                	#key1 = ("newDI_step_%03d%03d.csv" % (i+1,j+1))
                	name.append(key)
                	#name1.append(key1)
#std = ["STD_1F","STD_2F","STD_3F","STD_4F","STD_5F","STD_6F","STD_7F","STD_8F","STD_9F","STD_10F","STD_11F","STD_12F","STD_13F","STD_14F","STD_15F","STD_16F","STD_17F","STD_18F"]
#lbl = ["DI_1F","DI_2F","DI_3F","DI_4F","DI_5F","DI_6F","DI_7F","DI_8F","DI_9F","DI_10F","DI_11F","DI_12F","DI_13F","DI_14F","DI_15F","DI_16F","DI_17F","DI_18F"]
std = ['ACC_1FL','ACC_2FL','ACC_3FL','ACC_4FL','ACC_5FL','ACC_6FL','ACC_7FL','ACC_8FL','ACC_9FL','ACC_10FL','ACC_11FL','ACC_12FL','ACC_13FL','ACC_14FL','ACC_15FL','ACC_16FL','ACC_17FL','ACC_18FL','ACC_RFL']
lbl = ["maxDI_1F","maxDI_2F","maxDI_3F","maxDI_4F","maxDI_5F","maxDI_6F","maxDI_7F","maxDI_8F","maxDI_9F","maxDI_10F","maxDI_11F","maxDI_12F","maxDI_13F","maxDI_14F","maxDI_15F","maxDI_16F","maxDI_17F","maxDI_18F"]
#lbl1 = ['DI_1F','DI_2F','DI_3F','DI_4F','DI_5F','DI_6F','DI_7F','DI_8F','DI_9F','DI_10F','DI_11F','DI_12F','DI_13F','DI_14F','DI_15F','DI_16F','DI_17F','DI_18F']

# default = 180
pic_length = 360
s3 = boto3.client('s3')

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
        #print(d, len(d))
        return d/18

def check_func(x):
	if '111111111111111111' ==  x and random.random() > 0.02:
		return False
	else: 
		return True

#for n in name:
def d2_to_d3(arr):
	res=[]
	for x in arr:
		v1=[]
		for y in x:
			v1.append([y])
		res.append(v1)
	return res

datas = []
labels = []

pic_name = {}
pic_name['filename'] = []

for n in name:
	print(n)
	obj = s3.get_object(Bucket='takenaka', Key='add_data_20190426/'+n)
	#obj1 = s3.get_object(Bucket='takenaka', Key='newDI/'+n1)
	data = pd.read_csv(io.BytesIO(obj['Body'].read()))
	#label = pd.read_csv(io.BytesIO(obj1['Body'].read()))
	row_number = len(data.index)
	for i in range(int(row_number/pic_length)):
		if check_func("".join(map(str,data.iloc[i*pic_length:i*pic_length+pic_length][lbl1].max().tolist()))):
			pic = data.iloc[i*pic_length:i*pic_length+pic_length][std]
			pic = pic.values.tolist()
			pic_name['filename'].append(n[12:18]+str(i)+'_'+"".join(map(str,data.iloc[i*pic_length:i*pic_length+pic_length][lbl].max().tolist()))+'.jpg')
			datas.append(d2_to_d3(pic))
			labels.append(label_settings_3class("".join(map(str,data.iloc[i*pic_length:i*pic_length+pic_length][lbl].max().tolist()))))
			
			
np.save(path+"datas_%03d_%03d.npy" % (start+1, end+1), datas)
np.save(path+"labels_%03d_%03d.npy" % (start+1, end+1), labels)

pic_file=pd.DataFrame.from_dict(pic_name)
pic_file.to_csv('divided_by_20_pic_name_additional/pic_name_%03d_%03d.csv' % (start+1, end+1) ,index=False)


