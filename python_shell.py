import os
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

for i in range(start, end+1):
	print(i)
	os.system("python3 create_images_and_make_tensor_by_20.py %s %s" % (i, i))
