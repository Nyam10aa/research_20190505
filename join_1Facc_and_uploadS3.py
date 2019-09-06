import pandas as pd
from PIL import Image
import boto3
import io
import numpy as np
import random
import sys

import s3fs

s3 = boto3.client('s3')

for i in range(1,141):
	print(i)
	for j in range(1,101):#101):
		key = ("result_step_%03d%03d.csv" % (i, j ))
		obj = s3.get_object(Bucket='takenaka', Key=key)
		data = pd.read_csv(io.BytesIO(obj['Body'].read()))
		
		key_1f = ("ACC_1FL_%03d.csv" % i)
		obj_1f = s3.get_object(Bucket='takenaka', Key='ACCELERATION_DATA/ACC_1FL/'+key_1f)
		data_1f = pd.read_csv(io.BytesIO(obj_1f['Body'].read()))
		
		data_1f = data_1f * j
		
		res = pd.concat([data_1f, data], axis = 1)
		
		bytes_to_write = res.to_csv(None).encode()
		fs = s3fs.S3FileSystem(anon=False)#key=key, secret=secret)
		with fs.open("s3://takenaka/ACCELERATION_DATA/RAW_DATA_with_1FL_acc/result_step_%03d%03d.csv" % (i,j), 'wb') as f:
			f.write(bytes_to_write)
