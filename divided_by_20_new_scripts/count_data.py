import glob, os
import pandas as pd


pic_list = []

for file in sorted(glob.glob("./../divided_by_20_pic_name/*.csv")):
	d = pd.read_csv(file)
	print(file)
	print(d.shape)
	pic_list += list(d['filename'])
res_dic={}
for i in range(140):
	res_dic[i+1]={}
	for j in range(18):
		res_dic[i+1][j]={}
		for k in range(3):
			res_dic[i+1][j][k+1]=0
def five_to_three(i):
	x=int(i)
	if x==2:
		return x
	elif x==3:
		return x-1
	elif x==4:
		return x-2
	elif x==5:
		return x-2
	else:
		return x
for pic in pic_list:
	for j in range(18):
		res_dic[int(pic[0:3])][j][five_to_three(pic.split('.')[0].split('_')[1][j])]+=1
reform = {(level1_key, level2_key): values for level1_key, level2_dict in res_dic.items() for level2_key, values in level2_dict.items()}
df = pd.DataFrame(reform).T

df1=df.loc[1].T
for i in range(2,141):
	df1=pd.concat([df1, df.loc[i].T])
#df1.to_csv('result/shuukei.csv')
print(df1)

