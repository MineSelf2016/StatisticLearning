# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Chapter3'))
	print(os.getcwd())
except:
	pass

#%% [markdown]
# #  第3章 k近邻法
#%% [markdown]
# 1．$k$近邻法是基本且简单的分类与回归方法。$k$近邻法的基本做法是：对给定的训练实例点和输入实例点，首先确定输入实例点的$k$个最近邻训练实例点，然后利用这$k$个训练实例点的类的多数来预测输入实例点的类。
# 
# 2．$k$近邻模型对应于基于训练数据集对特征空间的一个划分。$k$近邻法中，当训练集、距离度量、$k$值及分类决策规则确定后，其结果唯一确定。
# 
# 3．$k$近邻法三要素：距离度量、$k$值的选择和分类决策规则。常用的距离度量是欧氏距离及更一般的**pL**距离。$k$值小时，$k$近邻模型更复杂；$k$值大时，$k$近邻模型更简单。$k$值的选择反映了对近似误差与估计误差之间的权衡，通常由交叉验证选择最优的$k$。
# 
# 常用的分类决策规则是多数表决，对应于经验风险最小化。
# 
# 4．$k$近邻法的实现需要考虑如何快速搜索k个最近邻点。**kd**树是一种便于对k维空间中的数据进行快速检索的数据结构。kd树是二叉树，表示对$k$维空间的一个划分，其每个结点对应于$k$维空间划分中的一个超矩形区域。利用**kd**树可以省去对大部分数据点的搜索， 从而减少搜索的计算量。
#%% [markdown]
# ### 距离度量
#%% [markdown]
# 设特征空间$x$是$n$维实数向量空间 ，$x_{i}, x_{j} \in \mathcal{X}$,$x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(n)}\right)^{\mathrm{T}}$,$x_{j}=\left(x_{j}^{(1)}, x_{j}^{(2)}, \cdots, x_{j}^{(n)}\right)^{\mathrm{T}}$
# ，则：$x_i$,$x_j$的$L_p$距离定义为:
# 
# 
# $L_{p}\left(x_{i}, x_{j}\right)=\left(\sum_{i=1}^{n}\left|x_{i}^{(i)}-x_{j}^{(l)}\right|^{p}\right)^{\frac{1}{p}}$
# 
# - $p= 1$  曼哈顿距离
# - $p= 2$  欧氏距离
# - $p= inf$   闵式距离minkowski_distance 

#%%
import math
from itertools import combinations


#%%
def L(x, y, p=2):
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0

#%% [markdown]
# ### 课本例3.1

#%%
x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]


#%%
#考虑距离度量中p为1到4的情况
for i in range(1, 5):
    r = {'最近点：{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    print(min(zip(r.values(), r.keys())))

#%% [markdown]
# # KNN实现
#%% [markdown]
# ## 1.导入必要的库

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

#%% [markdown]
# ## 2.导入数据集
#%% [markdown]
# #### 这里使用的是鸢尾属植物数据集，该数据集测量了所有150个样本的4个特征，分别是：
# #### sepal length（花萼长度）、sepal width（花萼宽度）、petal length（花瓣长度）、petal width（花瓣宽度）
# #### 此实验中，我们只考察前两个特征

#%%
#将导入的数据设置为DataFrame格式，并设置其列名
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']


#%%
df


#%%
# 绘制前两类数据（标签为0和1的数据）的散点图，（只考虑sepal length和sepal width这两个属性）
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

#%% [markdown]
# ## 3.数据集切分

#%%
# 只考虑数据的前两列属性以及最后一列的标签
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]

# 从样本中，随机按照80%和20%的比例抽取训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% [markdown]
# ## 4.实现KNN

#%%
class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数，即k值
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    # 预测未知类别的点X所属的类别
    def predict(self, X):
        # 先取出前n个点，分别计算X与这n个点的范数
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        # 对于n个点后面的所有点，计算其与X的范数，如果值小于上面前n个点范数的最大值，则进行替换，最后得到所有点中，距离X最近的n个点
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        # 分类决策，n个点中，多数点所属的类别即为预测所得的X所属类别
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count

    # 得分函数，用测试数据集进行测试，得到此方法的预测准确度
    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

#%% [markdown]
# ## 5.创建KNN实例

#%%
clf = KNN(X_train, y_train)

#%% [markdown]
# ## 6.模型准确率

#%%
print('准确率：{:.2%}'.format(clf.score(X_test, y_test)))

#%% [markdown]
# ## 7.使用新的测试点进行预测

#%%
test_point = [6.0, 3.0]
print('测试点X所属类别:{}'.format(clf.predict(test_point)))


#%%
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

#%% [markdown]
# ### scikit-learn实例

#%%
from sklearn.neighbors import KNeighborsClassifier


#%%
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)


#%%
clf_sk.score(X_test, y_test)

#%% [markdown]
# ### sklearn.neighbors.KNeighborsClassifier
# 
# - n_neighbors: 临近点个数
# - p: 距离度量
# - algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
# - weights: 确定近邻的权重
#%% [markdown]
# # kd树
#%% [markdown]
# k近邻法最简单的实现方法是线性扫描，这时要计算输入实例与每一个训练实例的距离，当训练集很大时，计算非常耗时，为了提高k近邻搜索的效率，可以考虑使用特殊的结构存储训练数据，以减少计算距离的次数，kd树就是其中的一种方法
# **kd**树是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。
# 
# **kd**树是二叉树，表示对$k$维空间的一个划分（partition）。构造**kd**树相当于不断地用垂直于坐标轴的超平面将$k$维空间切分，构成一系列的k维超矩形区域。kd树的每个结点对应于一个$k$维超矩形区域。
# 
# 构造**kd**树的方法如下：
# 
# 构造根结点，使根结点对应于$k$维空间中包含所有实例点的超矩形区域；通过下面的递归方法，不断地对$k$维空间进行切分，生成子结点。在超矩形区域（结点）上选择一个坐标轴和在此坐标轴上的一个切分点，确定一个超平面，这个超平面通过选定的切分点并垂直于选定的坐标轴，将当前超矩形区域切分为左右两个子区域
# （子结点）；这时，实例被分到两个子区域。这个过程直到子区域内没有实例时终止（终止时的结点为叶结点）。在此过程中，将实例保存在相应的结点上。
# 
# 通常，依次选择坐标轴对空间切分，选择训练实例点在选定坐标轴上的中位数
# （median）为切分点，这样得到的**kd**树是平衡的。注意，平衡的**kd**树搜索时的效率未必是最优的。
# 
#%% [markdown]
# ### 构造平衡kd树算法
# 输入：$k$维空间数据集$T＝\{x_1，x_2,…,x_N\}$，
# 
# 其中$x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(k)}\right)^{\mathrm{T}}$ ，$i＝1,2,…,N$；
# 
# 输出：**kd**树。
# 
# （1）开始：构造根结点，根结点对应于包含$T$的$k$维空间的超矩形区域。
# 
# 选择$x^{(1)}$为坐标轴，以T中所有实例的$x^{(1)}$坐标的中位数为切分点，将根结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴$x^{(1)}$垂直的超平面实现。
# 
# 由根结点生成深度为1的左、右子结点：左子结点对应坐标$x^{(1)}$小于切分点的子区域， 右子结点对应于坐标$x^{(1)}$大于切分点的子区域。
# 
# 将落在切分超平面上的实例点保存在根结点。
# 
# （2）重复：对深度为$j$的结点，选择$x^{(1)}$为切分的坐标轴，$l＝j(modk)+1$，以该结点的区域中所有实例的$x^{(1)}$坐标的中位数为切分点，将该结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴$x^{(1)}$垂直的超平面实现。
# 
# 由该结点生成深度为$j+1$的左、右子结点：左子结点对应坐标$x^{(1)}$小于切分点的子区域，右子结点对应坐标$x^{(1)}$大于切分点的子区域。
# 
# 将落在切分超平面上的实例点保存在该结点。
# 
# （3）直到两个子区域没有实例存在时停止。从而形成**kd**树的区域划分。
#%% [markdown]
# ## 1.构建kd树

#%%
# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split      # 整数（进行分割维度的序号）
        self.left = left        # 该结点分割超平面左子空间构成的kd-tree
        self.right = right      # 该结点分割超平面右子空间构成的kd-tree

# 构建kd树
class KdTree(object):
    def __init__(self, data):
        k = len(data[0])  # 数据维度

        # 创建结点
        def CreateNode(split, data_set):  # 按第split维划分数据集data_set创建KdNode
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])   # 将结点按照第split维进行排序
            split_pos = len(data_set) // 2          # //为Python中的整数除法
            median = data_set[split_pos]            # 中位数分割点
            split_next = (split + 1) % k            # 下一次进行分割的维度

            # 递归的创建kd树
            return KdNode(
                median,
                split,
                CreateNode(split_next, data_set[:split_pos]),      # 创建左子树
                CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preOrder(root):
    print(root.dom_elt, "split = ", root.split)
    print("*" * 50)
    if root.left:  # 节点不为空
        print(root.dom_elt, "的下一个左结点是", root.left.dom_elt)
        preOrder(root.left)
    else:
        print(root.dom_elt, "的下一个左结点是叶结点。")

    if root.right:
        print(root.dom_elt, "的下一个右结点是", root.right.dom_elt)
        preOrder(root.right)
    else:
        print(root.dom_elt, "的下一个右结点是叶结点。")



#%% [markdown]
# ### 例3.2

#%%
import pandas as pd 
import numpy as np 

# data = [[2, 3],[2, 1],[2, 4],[100, 0]]
df = pd.read_csv("Chapter3/datasets/test1.csv", header=None)
data = np.array(df)
data = data.tolist()

#%%
kd = KdTree(data)
preOrder(kd.root)

#%% [markdown]
# ## 2.搜索kd树

#%%
# 对构建好的kd树进行搜索，寻找与目标点最近的样本点：
from math import sqrt
from collections import namedtuple

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple","nearest_point  nearest_dist  nodes_visited")

# 搜索kd树，找出与point距离最近的点
def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"),0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split        # 进行分割的维度
        pivot = kd_node.dom_elt  # 结点

        #-----------------------------------------------------------------------------------
        #寻找point所属区域对应的叶结点
        if target[s] <= pivot[s]:         # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left    # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:                             # 目标离右子树更近
            nearer_node = kd_node.right   # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域
        print("temp1 = ", temp1)

        #-------------------------------------------------------------------------------------
        #以此叶节点作为“当前最近点”
        nearest = temp1.nearest_point  
        dist = temp1.nearest_dist  # 更新最近距离
        print("nearest: {}".format(nearest))
        print("dist: {}".format(dist))

        nodes_visited += temp1.nodes_visited
        print("nodes_visited: {}".format(nodes_visited))

        if dist < max_dist:
            max_dist = dist       # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])            # 第s维上目标点与分割超平面的距离
        print("超球体半径：{}".format(temp_dist))
        if max_dist < temp_dist:                         # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        #----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target)))
        print("temp_dist: {}".format(temp_dist))

        if temp_dist < dist:   # 如果“更近”
            nearest = pivot    # 更新最近点
            dist = temp_dist   # 更新最近距离
            max_dist = dist    # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)
        print("temp2: {}".format(temp2))

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:         # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point     # 更新最近点
            dist = temp2.nearest_dist         # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


#%%
ret = find_nearest(kd, [3, 1, 4])
print (ret)


#%%
from time import clock
from random import random

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]
 
# 产生n个k维随机向量 
def random_points(k, n):
    return [random_point(k) for _ in range(n)]     



#%%
N = 400000
t0 = clock()
kd2 = KdTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest(kd2, [0.1,0.5,0.8])      # 四十万个样本点中寻找离目标最近的点
t1 = clock()
print ("time: ",t1-t0, "s")
print (ret2)




#%%
