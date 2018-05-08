import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
def dataNormalize(dataset):
    m = dataset.shape[1]
    for i in range(m):
        dmin = dataset[:,i].min()
        dmax = dataset[:,i].max()
        if(dmax!=dmin):
            dataset[:,i] = (dataset[:,i]-dmin)/(dmax-dmin)
        else:
            dataset[:,i]= dataset[:,i]-dmin
    return dataset
def loadData(url):
    file = open(url,'r')
    tmp = np.loadtxt(file, delimiter=',')
    m=tmp.shape[1]#number of features
    labels = tmp[:,-1]
    dataset = tmp[:,:m-1]
    return labels,dataset

def chiSelect(dataset,labels):
    scoreList = []
    for i in range(5,101,5):
        model1 = SelectKBest(chi2,k=i)
        dataset1 = model1.fit_transform(dataset,labels)
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=1), dataset1, labels, cv=10)
        # print(scores)
        scoreList.append(scores.mean())
        np.savetxt('newmuskk.csv', scoreList, delimiter=',')
    return scoreList

#print(dataset.shape)
url = "E://pythonFile//musk2.csv"
labels,dataset = loadData(url)
#print(dataset[:,0])
dataset1 = dataNormalize(dataset)
l = chiSelect(dataset1,labels)

