import numpy as np
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
    n1 = 0
    n2 = 0
    f = list(zip(prob,labels))
    V = np.sum([value1 for value1, value2 in f if value2 == -1])
    rank = [value2 for value1, value2 in sorted(f, key=lambda x: x[0], reverse=True)]
    prob = [value1 for value1, value2 in sorted(f, key=lambda x: x[0], reverse=True)]
    if(calAUC1(prob,labels)>=0.5):
        for i in range(len(labels)):
            if(rank[i]==1):
                Rs1 = Rs1 + V - n1
                n2 = n2 + prob[i]
            else:
                n1 = n1 + prob[i]
                Rs2 = Rs2 + n2

    else:
        for i in range(len(labels)):
            if(rank[i]==-1):
                Rs1 = Rs1 + V - n1
                n2 = n2 + prob[i]
            else:
                n1 = n1 + prob[i]
                Rs2 = Rs2 + n2
    sAuc = 0
    if(negNum!=0 | posNum!=0):
        sAuc = (Rs2-Rs1)/(negNum*posNum)
    else:
        sAuc = 0
    return sAuc
prob=[0.95,0.89,0.2,0.16,0.15,0.13,0.1]
labels = [1,1,1,1,-1,-1,-1]
sauc = calsAUC(prob,labels)
print(sauc)
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
def rankFeature(url):
    labels,dataset = loadData(url)
    dataset = dataNormalize(dataset)
    s=0
    auclist=[]
    for prob in dataset.T:
        auc = calAUC1(prob,labels)
        auclist.append(auc)
    rankList = [values for values,values1 in sorted(list(enumerate(auclist)),key=lambda x:x[1],reverse=True)]
    return rankList
url = "E://pythonFile//musk2.csv"
rankList = rankFeature(url)
print(rankList)
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
scoreList = []
labels,dataset = loadData(url)
dataset = dataNormalize(dataset)
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
print(scoreList)