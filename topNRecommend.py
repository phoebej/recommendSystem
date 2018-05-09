
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from collections import Counter
_author_ = 'phoebe'
_project_= 'topNRecommend'

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)
user_num = df['user_id'].unique().shape[0]
item_num = df['item_id'].unique().shape[0]
data = defaultdict(set)
for line in df.itertuples():
    data[line[1]].add(line[2])
#    data[line[0]].add(line[1])
def SplitData(data,M,k,seed):
    test = []
    train = []
    random.seed(seed)
    for line in data.itertuples():
        if(random.randint(0,M)==k):
            test.append([line[0],line[1]])
        else:
            train.append([line[0],line[1]])
    return train,test
train,test = SplitData(df,8,1,0)
def GetRecommendation(user,n):
    None
def Recall(train,test,N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user,N)
        for item,pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit/(all*1.0)
def Coverage(train,test,N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user,N)
        for item,pui in rank:
            recommend_items.add(item)
    return len(recommend_items)/(len(all_items)*1.0)
def ItemSimilarity(data):
    #calculate
    N = defaultdict(int)
    C = defaultdict(defaultdict)
    for user,item in item_user.items():
        for i in item:
            N[i] += 1
            C.setdefault(i, defaultdict(int))
            for j in item:
                if(i == j):
                    continue
                C[i][j] += 1
    sim = defaultdict(defaultdict)
    for i,related_item in C.items():
        for j,cij in related_item.items():
            sim[i][j] = cij/(math.sqrt(N[i]*N[j]))
    print(sim)
    return sim
ItemSimilarity(data)



def UserSimilarity(data):
    #inverse table for item_users
    item_users = {}
    for user,item in data.items():
        for i in item:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(user)
    #calculate
    C = defaultdict(defaultdict)
    N = defaultdict(int)
    for item,user in item_users.items():
        for u in user:
            N[u]+=1
            C.setdefault(u,defaultdict(int))
            for v in user:
                if(u==v):
                    continue
                C[u][v] += 1
    sim = defaultdict(defaultdict)
    for u,related_user in C.items():
        for v,cuv in related_user.items():
            sim[u][v] = cuv/math.sqrt(N[u]*N[v])
    return sim
UserSimilarity(data)
def Recommend(user,train,sim):
    rank = dict()
    K = 3
    interact_items = train[user]
    for u,v in sorted(sim[user],key=lambda x:x[1],reverse=True)[0:K]:
        for i,rvi in train[u].items():
            if i in interact_items:
                rank[i] += rvi*v
    return rank
j



