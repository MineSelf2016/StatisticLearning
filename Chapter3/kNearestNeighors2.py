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
    def __init__(self, point, split, left, right):
        self.point = point  # k维向量节点(k维空间中的一个样本点)
        self.split = split      # 整数（进行分割维度的序号）
        self.left = left        # 该结点分割超平面左子空间构成的kd-tree
        self.right = right      # 该结点分割超平面右子空间构成的kd-tree

# 构建kd树
class KdTree:
    def __init__(self, data_list):
        # k = len(data[0])  # 数据维度
        dimensions = data_list.shape[1]  # 数据维度
        print("data_list: ", data_list)
        print("dimensions = ", dimensions)
        
        # 创建结点
        def CreateNode(data_set):  # 按第split维划分数据集data_set创建KdNode
            if len(data_set) == 0:  # 数据集为空
                print("划分结束")
                return 
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            # data_set.sort(key=lambda x: x[split])   # 将结点按照第split维进行排序
            # split_pos = len(data_set) // 2          # //为Python中的整数除法
            # median = data_set[split_pos]            # 中位数分割点
            # split_next = (split + 1) % k            # 下一次进行分割的维度
            vars = [data_set[:, i].var() for i in range(dimensions)]
            # 获取最大方差对应的维度
            print("vars = ", vars)
            split = np.argmax(vars)
            print("split = ", split)
            # 获取该split 维度上的中位数，该处做法不需要排序挪动整个数组，以免数据量增多时造成时间复杂度的极速上升。
            pivot = sorted(data_set[:, split])[data_set.shape[0] // 2]
            print("pivot = ", pivot)
            # 获取切分点
            point = data_set[np.where(data_set[:, split] == pivot)[0][0]]
            print("point = ", point)
            # 递归的创建kd树
            return KdNode(
                point,
                split,
                CreateNode(np.array(list(filter(lambda x : x[split] < pivot, data_set)))),      # 创建左子树
                CreateNode(np.array(list(filter(lambda x : ((x[split] >= pivot)  and not((x == point).all())), data_set))))) # 创建右子树

        
        self.root = CreateNode(data_list)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preOrder(root):
    print(root.point, "split = ", root.split)
    print("*" * 50)
    if root.left:  # 节点不为空
        print(root.point, "的下一个左结点是", root.left.point)
        preOrder(root.left)
    else:
        print(root.point, "的下一个左结点是叶结点。")

    if root.right:
        print(root.point, "的下一个右结点是", root.right.point)
        preOrder(root.right)
    else:
        print(root.point, "的下一个右结点是叶结点。")



#%% [markdown]
# ### 例3.2

#%%
import pandas as pd 
import numpy as np 

# data = [[2, 3],[2, 1],[2, 4],[100, 0]]
df = pd.read_csv("Chapter3/datasets/test1.csv", header=None)
data = np.array(df)
# data = data.tolist()

#%%
kd = KdTree(data)
print(kd.root)
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
        pivot = kd_node.point  # 结点
        print("pivot = ", pivot)
        print("- " * 30)
        print()
        #-----------------------------------------------------------------------------------
        #寻找point所属区域对应的叶结点
        if target[s] < pivot[s]:         # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left    # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:                             # 目标离右子树更近
            nearer_node = kd_node.right   # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域
        print("temp1 = ", temp1)

        print("- " * 30)
        print()
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
            print("不相交，继续向上回退。")
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        print("- " * 30)
        print()
        #----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        # 开方较占用内存，考虑使用乘方或者不开方进行比较
        temp_dist = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target)))
        print("temp_dist: {}".format(temp_dist))

        if temp_dist < dist:   # 如果“更近”
            print("当前分割点与目标点更近，准备更新nearest, dist 与 max_dist。")
            print()
            nearest = pivot    # 更新最近点
            dist = temp_dist   # 更新最近距离

            # 为什么这里更新，而下面另一个子结点遍历时却不更新超球体半径了？
            # max_dist 到底代表什么？
            max_dist = dist    # 更新超球体半径

        print("- " * 30)
        print("检查另一个子结点所有空间区域是否存在更近点")
        print("- " * 30)
        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)
        print("temp2: {}".format(temp2))

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:         # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point     # 更新最近点
            dist = temp2.nearest_dist         # 更新最近距离

        print("即将结束本次有效的搜索，返回上一层函数。")
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
