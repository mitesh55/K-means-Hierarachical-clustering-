import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# load dataset :
data  = pd.read_csv(r'E:\ExcelR ass\clustering\crime_data.csv')
# print(data.head())

# avoid sting column for mathematical operation :
df = data.iloc[:,1:]
# print(df.head())

# apply normalisation for scale down data values :
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_fun(df)

# declare cluster number and fit the model :
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_norm)
label = kmeans.fit_predict(df_norm)

# to find labels :
k_label = kmeans.labels_
# print(k_label)
# print(kmeans.labels_)

# append derived labels to main data set :
df["k_clustered"] = pd.Series(kmeans.labels_)
# print(df.head())
df_norm["k_clustered"] = pd.Series(kmeans.labels_)
# print(df_norm.head())

# to watch significant difference between each cluster :
result = df.groupby(kmeans.labels_).mean()
# print(result)

# to find cluster center :
cluster_cntr = kmeans.cluster_centers_
# print(cluster_cntr)
# print(label)

# to plot cluster :
plt.scatter(df_norm.Murder.loc[k_label==0], df_norm.Assault.loc[k_label==0],label="cluster-1", color="blue")
plt.scatter(df_norm.Murder.loc[k_label==1], df_norm.Assault.loc[k_label==1],label="cluster-1", color="red")
plt.scatter(df_norm.Murder.loc[k_label==2], df_norm.Assault.loc[k_label==2],label="cluster-1", color="green")

# plot cluster centers :
plt.scatter(cluster_cntr[0][0], cluster_cntr[0][1], marker='*', s=100, color="black")
plt.scatter(cluster_cntr[1][0], cluster_cntr[1][1], marker='*', s=100, color="black")
plt.scatter(cluster_cntr[2][0], cluster_cntr[2][1], marker='*', s=100, color="black")
plt.legend()
plt.show()


# to find best (optimum) k-value using Elbow method :
k = list(range(2,10))
twss = []
for i in k:
    new_kmeans = KMeans(n_clusters=i)
    new_kmeans.fit(df_norm)
    wss = []
    for j in range(i):
        wss.append(sum(cdist(df_norm.iloc[new_kmeans.labels_==j,:], new_kmeans.cluster_centers_[j].reshape(1, df_norm.shape[1]), 'euclidean')))
    twss.append(sum(wss))

# plot k vs twss graph to find best k-value :
# plt.plot(k, twss, 'ro-')
# plt.xlabel("k-values")
# plt.ylabel("total_within_ss")
# plt.show()

# print(df)