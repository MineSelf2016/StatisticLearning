"""
构建从数据集求感知机模型的例子

"""

#%%
import numpy as np 
import matplotlib.pyplot as plt 


#%%
class showPicture:  
    def __init__(self,data,w,b):  
        self.b = b  
        self.w = w  
        plt.figure(1)  
        plt.title('Perceptron classifier', size=14)  
        plt.xlabel('x-axis', size=14)  
        plt.ylabel('y-axis', size=14)  

        xData = np.linspace(0, 5, 100)  
        yData = self.expression(xData)  
        plt.plot(xData, yData, color='r', label='y1 data')  

        plt.scatter(data[0][0],data[0][1],s=50)  
        plt.scatter(data[1][0],data[1][1],s=50)  
        plt.scatter(data[2][0],data[2][1],marker='x',s=50,)  
        plt.savefig('2d.png',dpi=300)  
    def expression(self,x):  
        y = (-self.b - self.w[0]*x)/self.w[1]  
        return y  
    def show(self):  
        plt.show()  


#%%
class Perceptron:
    # 参数 a 为学习率
    def __init__(self, x, y, a = 1):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1], 1))
        self.b = 0
        self.a = 1

    
    def sign(self, w, b, x):
        result = np.dot(x, w) + b
        return int(result)


    def train(self):
        flag = True
        length = len(self.x)

        while flag:
            count = 0
            for i in range(length):
                tmpY = self.sign(self.w, self.b, self.x[i, :])
                # 误分类点
                if self.y[i] * tmpY <= 0:
                    count += 1
                    # 更新w , b
                    tmp = self.x[i, :].reshape(self.w.shape)
                    tmp = tmp.reshape(self.w.shape)
                    self.w = self.w + self.a * self.y[i] * tmp
                    self.b = self.b + self.a * self.y[i]
            if count == 0:
                flag = False

        return self.w, self.b


#%%
x1 = np.array((3, 3))
x2 = np.array((4, 3))
x3 = np.array((1, 1))
x = np.array([x1, x2, x3])
y1 = 1
y2 = 1
y3 = -1
y = np.array([y1, y2, y3])
w = np.zeros((x.shape[1], 1))
b = 0
a = 1

#%%
p = Perceptron(x, y)

w, b = p.train()


#%%
print(w)

print(b)

#%%
sf = showPicture(x, w, b)
sf.show()

#%%
