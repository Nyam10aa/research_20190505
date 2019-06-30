import os
import pandas as pd

for i in range(1,584):
	os.system("mv divided_by_20_additional/labels_%03d_%03d.npy divided_by_20/labels_%03d_%03d.npy"  % (i, i, i+140, i+140))
	print("mv divided_by_20_additional/datas_%03d_%03d.npy divided_by_20/datas_%03d_%03d.npy"  % (i, i, i+140, i+140))

#for i in range(1,584):

#for i in range(1,584):
#	d = pd.read_csv('divided_by_20_pic_name_additional/pic_name_%03d_%03d.csv' % (i, i))
#	def f(st):
#		return str('%03d' % (i+140)) +st[3:]
#	d['filename'] = d['filename'].apply(f)
#	d.to_csv('divided_by_20_pic_name/pic_name_%03d_%03d.csv' % (i+140, i+140), index = False)
