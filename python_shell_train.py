import os
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

for i in range(start, end+1):
	print(i)
	os.system("python train_keras_by_20.py %s" % i)
