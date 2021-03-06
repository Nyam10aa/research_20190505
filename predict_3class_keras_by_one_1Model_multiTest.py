from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam

import keras
import numpy as np
import pandas as pd
import sys
import boto3
import io

def read_from_s3(file_path, file_name):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket='takenaka', Key = file_path + file_name)
        return io.BytesIO(obj['Body'].read())

model_type = int(sys.argv[1])
startt = int(sys.argv[2])
endd = int(sys.argv[3])
output = sys.argv[4]

#end = start

model = keras.models.load_model("divided_by_20_keras_model/by_one/model_case_%d.h5" % model_type, compile=False)
model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

#x_test_total = np.load("divided_by_20/datas_%03d_%03d.npy" % (start, start))
#y_test_total = np.load("divided_by_20/labels_%03d_%03d.npy" % (start, start))

for start in range(startt, endd+1):
    x_test_total = np.load(read_from_s3('processed_data_STD/', 'datas_%03d_%03d.npy' % (start, start)))
    y_test_total = np.load(read_from_s3('processed_data_STD/', 'labels_%03d_%03d.npy' % (start, start)))
    print(x_test_total.shape)
    print(y_test_total.shape)


    # score = model.evaluate(x_test_total, y_test_total, verbose=0)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    pic_name = pd.read_csv("divided_by_20_pic_name/pic_name_%03d_%03d.csv" % (start, start))["filename"].tolist()
    pic_name1 = []
    end = start
    for i in range(len(pic_name)):
            for folder in range(start, end+1):
                    if pic_name[i].startswith("%03d" % folder):
                            pic_name1.append(pic_name[i])

    pic_name=pic_name1
    def find_pic_index_from_name_list(folder):
            res=[]
            for i in range(len(pic_name)):
                    if pic_name[i].startswith(folder):
                            res.append(i)
            #print(res)
            return res

    def emotion_to_vec(x):
        d = np.zeros(5)
        d[x] = 1.0
        return d

    def result_edit(lis):
            dic ={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[]}
            kai_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            result = []
            result1 = []
            for k in kai_list:
                    dic[k]=lis[k*3:k*3+3].tolist()
                    d = [0,0,0]
                    #print(dic[k])
                    d[(dic[k]).index(max(dic[k]))]=1.0/18
                    #print(d)
                    for i in d:
                            result.append(i)
            #print(result)

            for k in kai_list:
                    dic[k]=lis[k*3:k*3+3].tolist()
                    result1.append((dic[k]).index(max(dic[k])))
            return result1

    def result_to_label(lis):
            labl = ''
            for l in lis:
                    labl += str(l+1)
            return labl

    big_dic ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0}
    #big_dic_detailed = {0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}

    big_dic_detailed = {0:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},1:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},2:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},3:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},4:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},5:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},6:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},7:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},8:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},9:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},10:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},11:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},12:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},13:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},14:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},15:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},16:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},17:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]}}

    for folder in range(start, end+1):
            x_test = np.array([x_test_total[index] for index in find_pic_index_from_name_list("%03d" % folder)])
            y_test = np.array([y_test_total[index] for index in find_pic_index_from_name_list("%03d" % folder)])
            print("-------------------folder , length of x_test, y_test-------------------------")
            print(folder, len(x_test),len(y_test), type(x_test))
            if not len(x_test) is 0:
                    result = model.predict(x_test)
                    print(type(result))
            dic ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0}
            accuracy = 0
            dic_actual_detailed={0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}
            dic_actual_detailed={0:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},1:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},2:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},3:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},4:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},5:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},6:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},7:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},8:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},9:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},10:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},11:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},12:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},13:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},14:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},15:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},16:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},17:{0:[0,0,0],1:[0,0,0],2:[0,0,0]}}
            #dic_detailed = {0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}
            dic_detailed = {0:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},1:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},2:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},3:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},4:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},5:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},6:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},7:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},8:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},9:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},10:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},11:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},12:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},13:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},14:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},15:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},16:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},17:{0:[0,0,0],1:[0,0,0],2:[0,0,0]}}
            num_zero = 0
            labels_actual_and_predict = []
            for i in range(len(y_test)):
                    #print(find_pic_index_from_name_list("%03d" % folder))
                    la=np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))
                    labels_actual_and_predict.append(result_to_label(result_edit(y_test[i])))
                    labels_actual_and_predict.append(result_to_label(result_edit(result[i])))
                    dic[la]=dic[la]+1
                    big_dic[la]=big_dic[la]+1
                    if result_edit(y_test[i]) == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] and result_edit(result[i]) == result_edit(y_test[i]):
                            num_zero = num_zero + 1
                    for j in range(18):
                            dic_actual_detailed[j][result_edit(y_test[i])[j]][result_edit(y_test[i])[j]] += 1
                            dic_detailed[j][result_edit(y_test[i])[j]][result_edit(result[i])[j]] += 1
                            big_dic_detailed[j][result_edit(y_test[i])[j]][result_edit(result[i])[j]] += 1
                    if result_edit(result[i]) == result_edit(y_test[i]):
                            accuracy = accuracy + 1
                    else:
                            k=1
                            #print("                 id:"+str(i))
                            #print(result_edit(y_test[i]))
                            #print(result_edit(result[i]))
                            #print(np.array(result_edit(result[i]))-np.array(result_edit(y_test[i])))
                            #print("same label number")
                    #dic[np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))]=dic[np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))]+1
            #print(dic)
            #print(accuracy/len(y_test))
            #print(num_zero)
            #print(dic_detailed)
            #print(dic_actual_detailed)

            #---------------------------------------
            print("number by successful floor")
            for i in range(19):
                    print(str(i)+","+str(dic[i]))
            print("each floor by number")
            for i in range(18):
                    print("\n")
                    for j in range(3):
                            print(dic_detailed[i][j])
            print("each floor by percent")
            for i in range(18):
                    print("\n")
                    for j in range(3):
                            sum=float(np.sum(np.array(dic_detailed[i][j])))/100
                            if sum==0:
                                    print(list(np.array(dic_detailed[i][j])))
                            else:
                                    print(list(np.array(dic_detailed[i][j])/float(sum)))
            print("total by number and percent")
            table_dic_for_total ={0:np.array([0,0,0]),1:np.array([0,0,0]),2:np.array([0,0,0])}
            for i in range(18):
                    for j in range(3):
                            table_dic_for_total[j]=table_dic_for_total[j]+np.array(dic_detailed[i][j])
            for j in range(3):
                    print(list(table_dic_for_total[j]))
            for j in range(3):
                    print(list(100*table_dic_for_total[j]/float(np.sum(table_dic_for_total[j]))))
            print("total accuracy")
            matched=0
            alll=0
            for j in range(3):
                    matched=matched+table_dic_for_total[j][j]
                    alll=alll+np.sum(table_dic_for_total[j])
            print(matched/float(alll))
            with open(output, mode="a") as f:
		detail_info = []
		detail_info1 = []
		for j in range(3):
			detail_info += list(table_dic_for_total[j])
			detail_info1 += list(100*table_dic_for_total[j]/float(np.sum(table_dic_for_total[j])))
		detail_info = ",".join(map(str,detail_info))
		detail_info1 = ",".join(map(str,detail_info1))
    		f.write(str(model_type)+","+str(start)+","+str(matched/float(alll))+","+detail_info+","+detail_info1+"\n")#","+",".join(labels_actual_and_predict)+"\n")
            f.close()
    	#---------------------------------------
    print("big_dic and big_dic_detailed")
    print(big_dic)
    print(big_dic_detailed)
