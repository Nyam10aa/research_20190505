import boto3

def upload():
	path = './divided_by_20_keras_model/by_one/'
	#path = './divided_by_20/'
	s3 = boto3.resource('s3')
	bucket = s3.Bucket('takenaka')
	bucket_path = 'model_STD_1_723/'
	#bucket_path = 'processed_data_STD/'
	try:
		for i in range(1,724):
			file_stream = 'model_case_%d.h5' % i
			#file_stream = 'datas_%03d_%03d.npy' % (i, i)
			bucket.upload_file(path +file_stream, bucket_path + file_stream)
			print(path+file_stream)
	except:
		import traceback
		traceback.print_exc()
if __name__ == '__main__':
	upload()
