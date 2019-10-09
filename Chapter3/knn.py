"""
kd-tree 实现；
kd-tree 搜索。

"""

#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#%%
df = pd.read_csv("Chapter3/datasets/test1.csv", header=None)


#%%
data_list = np.array(df)


#%%
class kd_node:
    def __init__(self, point = None, split = None, left = None, right = None):
        self.point = point
        self.split = split
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.point)


def creat_kdtree(root, data_list):
    print("-" * 50)
    print("root = ", root)
    print("data_list = ", data_list)
    if len(data_list) == 0:
        print("划分结束")
        return 
    # 获取维度
    dimensions = data_list.shape[1]
    print("dimensions = ", dimensions)
    # 获取最大方差对应的维度
    vars = [data_list[:, i].var() for i in range(dimensions)]
    print("vars = ", vars)
    split = np.argmax(vars)
    print("split = ", split)
    # 获取该split 维度上的中位数，该处做法不需要排序挪动整个数组，以免数据量增多时造成时间复杂度的极速上升。
    pivot = sorted(data_list[:, split])[data_list.shape[0] // 2]
    print("pivot = ", pivot)
    # 获取切分点
    point = data_list[np.where(data_list[:, split] == pivot)[0][0]]
    print("point = ", point)
    root = kd_node(point, split)
    # 若存在某一坐标轴上相同的点，则可能导致错误。
    root.left = creat_kdtree(root.left, np.array(list(filter(lambda x : x[split] < pivot, data_list))))
    root.right = creat_kdtree(root.right, np.array(list(filter(lambda x : ((x[split] >= pivot)  and not((x == point).all())), data_list))))
    return root


#%%
root = None
tree = creat_kdtree(root, data_list)


#%%
def preorder(tree):
    print(tree.point, " split = ", tree.split)
    print("*" * 50)
    if tree.left:
        print(tree.point, "的下一个左结点是", tree.left.point)
        preorder(tree.left)
    else:
        print("下一个左结点是叶结点，当前结点是", tree.point)

    if tree.right:
        print(tree.point, "的下一个右结点是", tree.right.point)
        preorder(tree.right)
    else:
        print("下一个右结点是叶结点，当前结点是", tree.point)


preorder(tree)

#%%
def findNN(root, target):
    # 判断子结点是否是叶结点]
    temp_root = root
    nearst_neighbor = root
    min_dist = compute_dist(nearst_neighbor.point, target)
    node_list = []

    while temp_root:
        node_list.append(temp_root)
        split = temp_root.split
        dd = compute_dist(temp_root.point, target)
        print("temp root: ", temp_root.point)
        print("dd: ", dd)
        if min_dist > dd:
            nearst_neighbor = temp_root
            min_dist = dd
        # 此处的等号是因为在构建kd tree 的时候，将与枢纽元相同的值分配到了右子树 ??????
        if target[split] < temp_root.point[split]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right

    print("node_list: ")
    for i, node_i in enumerate(node_list):
        print(node_i, end=", ")
    print()

    nearst_neighbor = None
    min_dist = float("inf")

    while node_list:
        # 递归向上搜索
        temp_root = back_point = node_list.pop()
        split = back_point.split

        # 判断当前结点是否是“当前最近点”
        cur_dist = compute_dist(temp_root.point, target)
        if cur_dist < min_dist:
            min_dist = cur_dist
            nearst_neighbor = temp_root
            print("当前最近点：", nearst_neighbor.point)
            print("当前最近距离：", min_dist)

        # 判断超球体是否与超矩形相交
        if abs(target[split] - back_point.point[split]) < min_dist:
            # 进入另一子空间进行搜索
            if target[split] < back_point.point[split]:
                temp_root = back_point.right
                if temp_root:
                    print("进入右子树: {}".format(back_point.right.point))
                
            else:
                temp_root = back_point.left
                if temp_root:
                    print("进入左子树: {}".format(back_point.left.point))
            # 此处的append() 仅仅将父结点的另外一个子结点自身加入到了队列，而并没有将其对应的子空间内的所有点加入进来，故会出错！
            if temp_root:
                node_list.append(temp_root)
                # cur_dist = compute_dist(temp_root.point, target)
                # if min_dist > cur_dist:
                #     # print(min_dist)
                #     min_dist = cur_dist
                #     # print(temp_root)
                #     nearst_neighbor = temp_root

    return nearst_neighbor.point, min_dist
    


#%%
def compute_dist(pt1, pt2):
    return np.linalg.norm(pt1-pt2)


#%%
nn, min_dist = findNN(tree, np.array([3, 1, 4]))

#%%
nn

2,4,5
6,1,4
1,4,4
0,5,7
5,2,5
4,0,6
7,1,6

#%%
min_dist

#%%
