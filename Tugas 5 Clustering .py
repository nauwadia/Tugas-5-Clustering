import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Buat dataframe tabel
data = {
    'Nama Kelas':['10 TKJ 1', '10 TKJ 2', '10 TKJ 3', '10 RPL', '10 TBSM 1', '10 TBSM 2','11 TKJ 1','11 TKJ 2','11 TKJ 3','11 TBSM', '11 RPL','12 TKJ 1','12 TKJ 2','12 RPL','12 TBSM'],
    'Sarana Penunjang':[7,16,13,20,14,13,21,18,24,27,22,18,22,26,17],
    'Stabilitas Jaringan':[2,2,1,3,3,4,6,20,8,6,1,2,3,1,1],
    'Nilai Rata-rata':[10,10,8,8,28,18,19,29,27,40,0,14,12,4,29]
}

df = pd.DataFrame(data)

# Ambil hanya kolom 'Sarana Penunjang','Stabilitas Jaringan','Nilai Rata-rata' untuk clustering
X = df.drop(columns=['Nama Kelas'])

# Buat model KMeans dengnan 2 cluster
Centroids = np.array([
   #centroid 1
    [7,2,10], 
    #centroid 2
    [27,6,40]
])

kmeans = KMeans (n_clusters=2, init=Centroids, n_init=1, random_state=0)
df['Cluster']= kmeans.fit_predict(X)

# Tampilkan hasil clustering
print('===> HASIL CLUSTERING <===')
print (df[['Nama Kelas', 'Cluster']])


