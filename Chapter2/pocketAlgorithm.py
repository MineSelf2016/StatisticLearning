# -*- encoding:utf-8 -*-

#%%
from numpy import vectorize
import numpy as np
import matplotlib.pyplot as plt
 
class Pocket:
 
    def __init__(self,random_state=None):
        self.numberOfIter=7000#最大迭代次数
        self.minWeights=None
        self.intercept=None#bias 截距
        self.errorCountArr=np.zeros(7000)#统计每次迭代的出错数
        self.errorCount=[]#统计每次优化或者每个更新权值时候的出错次数
        self.minErrors=7000#出错数量的初始值，一开始一般设置一个比较大的值
        self.random_state=random_state
 
    def predict(self,z):
        if z<0:
            return -1
        else:
            return 1
 
    def checkPredictedValue(self,z,actualZ):
        if(z==actualZ):
            return True
        else:
            return False
 
    def fit(self,X,Y):
        row,col=X.shape
        weights=np.array([1.0,1.0,1.0,1.0])
        vpredict = vectorize(self.predict)
        vcheckPredictedValue=vectorize(self.checkPredictedValue)
        learning_rate=1.0
        bias_val=np.ones((row,1))
        data=np.concatenate((bias_val,X),axis=1)
        np.random.seed(self.random_state)
        count=0
        iter=0
        while self.numberOfIter>0:
            weightedSum=np.dot(data,weights)
            predictedValues=vpredict(weightedSum)
            predictions=vcheckPredictedValue(predictedValues,Y)
            misclassifiedPoints=np.where(predictions==False)#分类错误的数据
            misclassifiedPoints=misclassifiedPoints[0]
            numOfErrors=len(misclassifiedPoints)#分类错误的数据量
            self.errorCountArr[iter]=numOfErrors
            if numOfErrors<self.minErrors:
                self.minErrors=numOfErrors
                self.errorCount.append(self.minErrors)
                count+=1
            iter+=1
            misclassifiedIndex=np.random.choice(misclassifiedPoints)#这一步与PLA不同，
                                                                    # 在此是随机从错误数据点里面选择一个点，进行更新权值
            weights+=(Y[misclassifiedIndex]*learning_rate*data[misclassifiedIndex])
            self.numberOfIter-=1
        self.weights=weights[1:]
        self.intercept=weights[0]
 
def main():
    data=np.loadtxt('classification.txt',dtype='float',delimiter=',',usecols=(0,1,2,4))
    X=data[:,0:3]
    Y=data[:,3]
    p=Pocket(random_state=2308863)
    p.fit(X,Y)
    print("Weights:")
    print(p.weights)
    print("Intercept Value:")
    print(p.intercept)
    print("Minimum Number Of Errors:")
    print(p.minErrors)
    ax1=plt.subplot(121)
    ax1.plot(np.arange(0,7000),p.errorCountArr)
    ax2=plt.subplot(122)
    ax2.plot(np.arange(0,len(p.errorCount)),p.errorCount)
    plt.show()
 
if __name__ == "__main__":
    main()

#%%
