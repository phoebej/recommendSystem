import xgboost as xb
#from sklearn.datasets.samples_generator import make_classification
#X1, Y1 = make_classification(n_samples=400000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)
#param={}
from sklearn import feature_selection
import numpy as np
import math
import matplotlib.pyplot as plt
def loadData(url):
    file = open(url,'r')
    tmp = np.loadtxt(file, delimiter=',')
    m=tmp.shape[1]#number of features
    labels = tmp[:,-1]
    dataset = tmp[:,:m-1]
    return labels,dataset
def dataNormalize(dataset):
    m = dataset.shape[1]
    for i in range(m):
        dmin = dataset[:,i].min()
        dmax = dataset[:,i].max()
        dataset[:,i] = (dataset[:,i]-dmin)/(dmax-dmin)
    return dataset
def calAUC1(prob,labels):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    #print(auc)
    return auc
def calsAUC(prob,labels):
    negNum = 0
    posNum = 0
    for i in range(len(labels)):
        if(labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    Rs1 = 0
    Rs2 = 0
    V = 0
    n0 = 0
    n1 = 0
    f = list(zip(prob,labels))
    if(calAUC1(prob,labels)>=0.5):
        V = np.sum([prob for value1,value2 in f if value2==-1])
        print(V)
def calAUC(prob,labels):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    if(auc<0.5):
        auc = 1 - auc
    #print(auc)
    return auc
    #if(auc<0.5)
def rankFeature(url):
    labels,dataset = loadData(url)
    s=0
    auclist=[]
    for prob in dataset.T:
        auc = calAUC(prob,labels)
        auclist.append(auc)
    rankList = [values for values,values1 in sorted(list(enumerate(auclist)),key=lambda x:x[1],reverse=True)]
    return rankList
def wronginsec(f1,f2,labels):
    p = len(labels[labels==1])
    n = len(labels[labels==-1])
    Y1 = []
    Y2 = []
    if (calAUC1(f1, labels) >= 0.5):
        #Y1 = np.concatenate((-1*np.ones((n,1)),np.ones((p,1))))
        Y1 = [-1 for i in range(n)]+[1 for i in range(p)]
    else:
        #Y1 = np.concatenate((np.ones((p,1)),-1*np.ones((n,1))))
        Y1 = [1 for i in range(p)] + [-1 for i in range(n)]
    if (calAUC1(f2, labels) >= 0.5):
        Y2 = [-1 for i in range(n)] + [1 for i in range(p)]
    else:
        Y2 = [1 for i in range(p)] + [-1 for i in range(n)]
    f = list(zip(f1, labels))
    y1 = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    inx1 = np.argsort(f1)
    inx2 = np.argsort(f2)
    f = list(zip(f2, labels))
    y2 = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    w1 = inx1[[i for i in range(len(y1)) if y1[i]!=Y1[i]]]
    w2 = inx2[[i for i in range(len(y2)) if y2[i]!=Y2[i]]]
    #res1 = [i for i in w1 if i in w2]
    res = list(set(w1).intersection(set(w2)))
    index = len(res)/(w1.shape[0]+w1.shape[0])
    return res,index

def findk(labels,k,alpha):
    #the order of return values is positive:negative
    #alpha>=k+1
    p = len(labels[labels == 1])
    n = len(labels[labels == -1])
    k1 = min(alpha+math.floor((k*p)/max(n,p)),k,p)
    k2 = min(alpha + math.floor((k*n)/max(n,p)),k,n)
    return k1,k2

def calNN(wronginstance,data,k1,k2,labels):
    #calculate the adaptive k nearest neigbors each classes
    #wronginstance is one of the common misclassified instance
    #data is the feature space of two features
    p = len(labels[labels == 1])
    n = len(labels[labels == -1])
    datasetsize = data.shape[0]
    for i in range(datasetsize):
        if(np.all(wronginstance==data[i])):
            s = i
            y1 = labels[i]
    dataset = np.delete(data,s,axis=0)
    diffMat = np.tile(wronginstance,(datasetsize-1,1)) - dataset
    sqdiffMat = diffMat**2
    distance = (sqdiffMat.sum(axis = 1))**0.5
    sortIndex = np.argsort(distance)
    y = labels[list(sortIndex)]
    hit = []
    miss = []
    for i in sortIndex:
        if(labels[i]*y1>0):
            hit.append(i)
        else:
            miss.append(i)
    if(y1==1):
        hit = hit[0:k1]
        miss=miss[0:k2]
    else:
        hit = hit[0:k2]
        miss = miss[0:k1]
    # return index of nearest hits and misses
    #the number of hits and misses are chosen adaptively
    disHit = 0
    disMiss = 0
    for i in hit:
        diff = (wronginstance-dataset[i])**2
        disHit += (diff.sum())**0.5
    for i in miss:
        diff = (wronginstance-dataset[i])**2
        disMiss +=(diff.sum())**0.5
    com = float(disMiss/len(miss)-(disHit/len(hit)))
    return com
url = "Colon.csv"
labels,dataset = loadData(url)
dataset = dataNormalize(dataset)
#print(labels[np.argsort(dataset[:,1])])
n = dataset.shape[0]
m = dataset.shape[1]
complementarity = np.zeros((m,m),dtype=float)


for i in range(m):
    f1 = dataset[:,i]
    for j in range(m):
        com = 0
        f2 = dataset[:,j]
        w,index=wronginsec(f1,f2,labels)
        newdataset = np.concatenate((f1[w][:,np.newaxis],f2[w][:,np.newaxis]),axis=1)
        data1 = np.concatenate((f1[:,np.newaxis],f2[:,np.newaxis]),axis=1)
        k1,k2 = findk(labels,8,11)
        for line in newdataset:
            com += calNN(line,data1,k1,k2,labels)
        if(i!=j):
            complementarity[i,j] = com/len(w)
        else:
            complementarity[i,j] = 0
       # com[i][j] = c


print(complementarity)
#print(np.concatenate((dataset[:,55][f][:,np.newaxis],dataset[:,33][f][:,np.newaxis]),axis=1))

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset[:,2],dataset[:,1],15.0*labels,15.0*labels)
plt.show()
#print(rankList)
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
scoreList = []
clf = svm.SVC()
r = rankList[:2]
data=dataset[:,r]
#print(data)
s = cross_val_score(svm.SVC(kernel='linear'),dataset,labels,cv=10,scoring="accuracy")
#print(s)
for i in range(5,101,5):
    r = rankList[:i]
    data = dataset[:,r]
    #clf = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(svm.SVC(kernel='linear'),data,labels,cv=10)
    #print(scores)
    scoreList.append(scores.mean())
#print(scoreList)
#print(scores)
#print(len(scores))
#def Classify():
#c1 = Solution(l,labels)
#auc = c1.calAUC()
'''