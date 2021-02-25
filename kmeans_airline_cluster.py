import pandas as pd
from sklearn.cluster import  KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# load dataset :
data  = pd.read_csv(r'E:\ExcelR ass\clustering\airline_final.csv')
# print(data.head(50))

# convert larger scale data into normalize form between (0-1)
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df = norm_func(data)


# initiate kmeans clustering parameter :
kmeans_clust = KMeans(n_clusters=5)

# fit model :
kmeans_clust.fit(df)

# to get labels :
# print(kmeans_clust.labels_)

# assign obtain cluster group to individual data on main dataset :
data["k_clustered"] = pd.Series(kmeans_clust.labels_)

# to see significant difference between clustered group :
results = data.groupby(data.k_clustered).mean()
# print(results)

# to find best (optimum)  k-value :
k = list(range(2,9))
twss = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    wss = []
    for j in range(i):
        wss.append(sum(cdist(df.iloc[kmeans.labels_==j,:], kmeans.cluster_centers_[j].reshape(1, df.shape[1]), 'euclidean')))
    twss.append(sum(wss))

# plt.plot(k, twss, 'ro-')
# plt.xlabel("no of clusters")
# plt.ylabel("total_within_sum_of_square")
# plt.show()



# to visulize cluster we take only two columns for ease for understanding :
# extract two features from row dataset for better understanding :
x = data.iloc[:,[1,6]]
# print(x.head())

# aplly normlization and fit model as above :
x_norm = norm_func(x)
x_kmeans = KMeans(n_clusters=4)
x_kmeans.fit(x_norm)
# print(x_kmeans.labels_)
new_k_label = x_kmeans.labels_
x_norm["k_clustered"] = pd.Series(x_kmeans.labels_)

# to find cluster center :
cluster_cntr = x_kmeans.cluster_centers_
# print(cluster_cntr)
# print(x.head(50))
# plt.scatter(x)
# print(x_norm.head(250))

# plot cluster obtained after applying two features to model :
# plt.scatter(x_norm.Balance.loc[new_k_label==0], x_norm.Bonus_miles[new_k_label==0], label="cluster-1", color="red")
# plt.scatter(x_norm.Balance.loc[new_k_label==1], x_norm.Bonus_miles[new_k_label==1], label="cluster-2", color="yellow")
# plt.scatter(x_norm.Balance.loc[new_k_label==2], x_norm.Bonus_miles[new_k_label==2], label="cluster-3", c="blue")
# plt.scatter(x_norm.Balance.loc[new_k_label==3], x_norm.Bonus_miles[new_k_label==3], label="cluster-4", c="green")

# plot cluster center :
# plt.scatter(cluster_cntr[0][0], cluster_cntr[0][1], marker="*", s=100, color="black")
# plt.scatter(cluster_cntr[1][0], cluster_cntr[1][1], marker="*", s=100, color="black")
# plt.scatter(cluster_cntr[2][0], cluster_cntr[2][1], marker="*", s=100, color="black")
# plt.scatter(cluster_cntr[3][0], cluster_cntr[3][1], marker="*", s=100, color="black")
# plt.legend()
# plt.show()

# to see significant difference between each cluster :
new_results = x_norm.groupby(x_norm.k_clustered).mean()
# print(new_results)
