"""
感知机模型：
    1、训练数据样本；
    2、选取初始值；
    3、判断是否为误分类点；
    4、若是误分类点，则随机梯度下降，并更新权值 w 与 偏置值 b；
    5、直至无误分类点。

"""

#%%
import numpy as np 
import matplotlib.pyplot as plt 


#%%
class showPicture:
    def __init__(self, xSample, ySample, w, b):
        self.w = w
        self.b = b
        self.xData = np.linspace(-2, 2, 100)
        self.yData = self.expression()
        plt.figure(1)
        plt.title("Perception classifier", size = 14)
        plt.xlabel("x-axis", size = 14)
        plt.ylabel("y-axis", size = 14)
        plt.xlim(left = -2, right = 2)
        plt.ylim(bottom = -2, top = 2)

        for i, xi in enumerate(xSample):
            plt.scatter(xi[0], xi[1], s = 50, color = "b" if y[i] == 1 else "r", marker = "o" if y[i] == 1 else "x")
        

    def expression(self):
        yData = (-self.b - self.w[0] * self.xData) / self.w[1]
        return yData


    def getCenterPoint(self):
        centerX = (self.xData[0] + self.xData[-1]) / 2
        centerY = (self.yData[0] + self.yData[-1]) / 2
        return centerX, centerY


    def drawNormalVector(self, A, B, x0, y0):
        xData = np.linspace(x0, x0 + 0.25, 20)
        yData = (B / A) * xData + (y0 - B / A * x0)
        plt.plot(xData, yData, color = "r", label = "normal vector")

    def show(self, *errorPoint):
        # print(errorPoint[0])
        if len(errorPoint) != 0:
            plt.scatter(errorPoint[0][0], errorPoint[0][1], s = 50, color = "k")
            plt.text(errorPoint[0][0] + 0.2, errorPoint[0][1] + 0.2, "Error Point(" + str(errorPoint[0][0]) + ", " + str(errorPoint[0][1]) + ")")
        line = ""
        for i, wi in enumerate(self.w):
            line += str(wi[0]) + "* x" + str(i) + " + " 
        line += str(self.b) + " = 0"
        plt.plot(self.xData, self.yData, color = "k", label = "y1 data")
        # 获取直线中点坐标，用于添加备注与法向量
        x0 = self.getCenterPoint()[0]
        y0 = self.getCenterPoint()[1]
        self.drawNormalVector(self.w[0][0], self.w[1][0], x0, y0)
        plt.text(x0, y0, line)
        plt.show()


#%%
class perceptron:
    def __init__(self, x, y, a = 1):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1], 1))
        self.b = 0
        # a 为学习率
        self.a = a


    def sign(self, w, b, x):
        return np.dot(x, w) + b


    def SGD(self, *args):
        flag = True
        while flag:
            # count 用于判断是否全部点都已正确分类，然后跳出循环
            count = 0
            for i, xi in enumerate(self.x):
                # 判断该点是否是误分类点
                tmpY = self.sign(self.w, self.b, xi)
                if self.y[i] * tmpY <= 0:
                    # 更新权值 w 与 偏置值 b
                    count += 1
                    # 作图
                    sp = showPicture(self.x, self.y, self.w, self.b)
                    sp.show(xi)

                    deltaW = self.a * self.y[i] * xi
                    deltaB = self.a * self.y[i]
                    # print("deltaB = ", deltaB)
                    # if abs(deltaB) < self.a:
                    #     flag = False
                    self.w = self.w + deltaW.reshape(self.w.shape)
                    self.b = self.b + deltaB
            if count == 0:
                flag = False
        return self.w, self.b


#%%
# x1 = np.array((-0.8, 0.9))
# x2 = np.array((-0.75, 0.6))
# x3 = np.array((-0.6, 0.7))
# x4 = np.array((-0.4, 0.75))
# x5 = np.array((-0.2, -0.2))

# x6 = np.array((0.25, 0.65))
# x7 = np.array((0.35, -0.5))
# x8 = np.array((0.4, 0.4))
# x9 = np.array((0.6, -0.9))
# x10 = np.array((0.9, 0.8))


# y1 = y2 = y3 = y4 = y5 = 1
# y6 = y7 = y8 = y9 = y10 = -1

# x = np.array((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10))
# y = np.array((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10))

x1 = np.array((0, 0))
x2 = np.array((1, 1))
x3 = np.array((0, 1))
x4 = np.array((1, 0))

y1 = y2 = 1
y3 = y4 = -1

x = np.array((x1, x2, x3, x4))
y = np.array((y1, y2, y3, y4))


#%%
p = perceptron(x, y)
w, b = p.SGD()
print("w = ", w)
print("b = ", b)


#%%
sp = showPicture(x, y, w, b)
sp.show()


#%%
