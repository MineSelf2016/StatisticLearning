

#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KDTree

#%%
df = pd.read_csv("Chapter3/datasets/test2.csv", header=None)

#%%
data_list = np.array(df)
X = np.array(data_list)

#%%
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X)


#%%
distances, indices = nbrs.kneighbors(X)
indices

#%%
distances

#%%
nbrs.kneighbors_graph(X).toarray()

#%%
tree = KDTree(X, leaf_size=2)    
dist, ind = tree.query(X, k = 2)



#%%
dist

#%%
ind

#%%
