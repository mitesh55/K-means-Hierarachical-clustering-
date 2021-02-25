import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# load dataset :
data  = pd.read_csv(r'E:\ExcelR ass\clustering\crime_data.csv')
# print(data.head())

# avoid sring format data for mathematical operation :
df = data.iloc[:,1:]
# print(df.head())

# scale down data for better understanding using normalisation :
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_fun(df)
# print(df_norm.head())


# plot dendogram to find howmany cluster is good for data :
dendogram = sch.dendrogram(sch.linkage(df_norm, method='complete'))
# plt.show()

# to apply Agglomerative clustering and fit the model :
hc = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean')
y_hc = hc.fit_predict(df_norm)
# print(y_hc)

# assign cluster label to row dataset :
data["h_clustered"] = pd.Series(y_hc)
df["h_clustered"] = pd.Series(y_hc)
# print(data.head())

# to see the significant difference betweence between each cluster :
result = df.groupby(df.h_clustered).mean()
# print(result)