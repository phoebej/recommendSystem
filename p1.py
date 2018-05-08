import xgboost as xb
#from sklearn.datasets.samples_generator import make_classification
#X1, Y1 = make_classification(n_samples=400000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)
#param={}
from sklearn import feature_selection
import numpy as np
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
url = "E://pythonFile//Colon.csv"
rl = rankFeature(url)
#print(rl)
labels,dataset = loadData(url)
dataset = dataNormalize(dataset)
rankList = rankFeature(url)
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