import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# load dataset :
data  = pd.read_csv(r'E:\ExcelR ass\clustering\airline_final.csv')
# print(data.head(50))

# scale down scaler data :
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df = norm_func(data)
# print(df.head(50))

# to plot dendogram :
# dendogram = sch.dendrogram(sch.linkage(df, method='complete'))
# plt.show()

# to apply Agglomerative Clustering and fit the model :
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
y_hc = hc.fit_predict(df)
# print(y_hc)

# assign derived cluster values to each data on main dataset :
data["h_clustered"] = pd.Series(y_hc)
# print(data.head())

# to see significant differencee between each cluster's feature :
results = data.groupby(data.h_clustered).mean()
print(results)