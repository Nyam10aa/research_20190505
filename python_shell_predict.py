import os
import sys

#start = int(sys.argv[1])
#end = int(sys.argv[2])

#------------1Model multiTest------------------
# for model in range(1,724):
# 	#for startt in range(1,723):
# 	print(model)
# 	os.system("time python predict_3class_keras_by_one_1Model_multiTest.py %d %d %d divided_by_20_results/result_0702_723x723.csv" % (model, 1, 723))


#------------multiModel 1Test------------------
for start in range(1,724):
	#for startt in range(1,723):
	print(start)
	os.system("time python predict_3class_keras_by_one_1Model_multiTest.py %d %d divided_by_20_results/result_0711_723x723.csv" % (start, start))