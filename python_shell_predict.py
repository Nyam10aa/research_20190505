import os
import sys

#start = int(sys.argv[1])
#end = int(sys.argv[2])

for model in range(1,2):
	#for startt in range(1,723):
	print(model)
	os.system("time python predict_3class_keras_by_one.py %d %d %d divided_by_20_results/result_0618_2.csv" % (model, 1, 30))
