import math
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)
user_num = df['user_id'].unique().shape[0]
item_num = df['item_id'].unique().shape[0]
train_data,test_data = cv.train_test_split(df,test_size=0.25)#split data set
train_data_matrix = np.zeros((user_num,item_num))
test_data_matrix = np.zeros((user_num,item_num))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
#print(train_data_matrix)
# user-item matrix
def calSimilarity(data_matrix,type='user'):
    if(type =='user'):
        simMatrix = np.zeros((user_num, user_num))
        for i in range(user_num):
            for j in range(i,user_num):
                #if(np.sum(data_matrix[i,:])*np.sum(data_matrix[j,:]!=0):
                simMatrix[i,j] = (np.sum(np.multiply(data_matrix[i,:],data_matrix[j,:])))\
                                 /(np.sum(data_matrix[i,:]**2)*(np.sum(data_matrix[j,:]**2)))
    else:
        simMatrix = np.zeros((item_num, item_num))
        for i in range(item_num):
            for j in range(i,item_num):
                simMatrix[i,j] = (np.sum(np.multiply(data_matrix[:,i],data_matrix[:,j])))\
                                 /(np.sum(data_matrix[:,i]**2)*(np.sum(data_matrix[:,j]**2)))
    return simMatrix
sim = calSimilarity(train_data_matrix,'user')
def predict(similarity,rating,type='user',k=3):
    if(type=='user'):
        mean_score = rating.mean(axis=1)
        rating_diff = (rating - mean_score[:,np.newaxis])
        pred = mean_score[:,np.newaxis] + similarity.dot(rating_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
       # pred = mean_score[:, np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    print(pred)
    return pred
predict(sim,train_data_matrix)





#print(sim)





def RMSE(records):
    return math.sqrt(sum([(rui-pui)*(rui-pui) for u,i,rui,pui in records])/float(len(records)))
def MAE(records):
    return sum([abs(rui-pui) for u,i,rui,pui in records])/float(len(records))


