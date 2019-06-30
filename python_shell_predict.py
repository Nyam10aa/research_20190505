import os
import sys

#start = int(sys.argv[1])
#end = int(sys.argv[2])

for model in range(2,3):
	for case in range(1,41):
		print(model, case)
		os.system("time python predict_3class_keras_by_one.py %d %d divided_by_20_results/result_0618_2.csv" % (model, case))
