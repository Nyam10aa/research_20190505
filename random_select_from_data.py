import pandas as pd
import csv


pic_name=pd.read_csv("pic_name.csv")




count={}
count1={}
for i in range(140):
	count["%03d" % (i+1)]=0
	count1["%03d" % (i+1)]=0
                
for p in pic_name["filename"]:
	if p[-22:-4] == '111111111111111111':
		count[p[0:3]] += 1
		count1[p[0:3]] += 18


result = []
for p in pic_name["filename"]:
	if p[-22:-4] == '111111111111111111':
		if random.randint(1,100) <= 20:
			result.append([p])
	else:
		result.append([p])


with open("pic_name.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(result)

