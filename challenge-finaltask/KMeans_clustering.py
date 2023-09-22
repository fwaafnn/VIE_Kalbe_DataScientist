# %% [markdown]
# ##### Afwa Afini - VIE Data Scientist - Kalbe Nutritionals
# ```
# Tujuan dari pembuatan model machine learning ini adalah untuk dapat membuat cluster customer-customer yang mirip

# %% [markdown]
# ##### Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# library untuk machine learning model
from sklearn.cluster import KMeans 

# %% [markdown]
# ##### Exploratory Data Analysis

# %%
# Membaca data customer
df_cs = pd.read_csv('../data-source/DS-Challenge-Customer.csv')
df_cs

# %%
df_cs.info()

# %%
# Membaca data transaction
df_tr = pd.read_csv('../data-source/DS-Challenge-Transaction.csv')
df_tr

# %%
df_tr.info()

# %% [markdown]
# ##### Data Preprocessing

# %%
df_cs.isna().sum()

# %%
# imputasi missing value
df_cs['marital_status'].fillna('Other', inplace=True)

# %%
df_cs['marital_status'].value_counts()

# %%
df_cs.isna().sum()

# %% [markdown]
# ##### Merge data

# %%
df_merge = df_cs.merge(df_tr, how='inner', on='CustomerID')
df_merge

# %% [markdown]
# ##### Drop data

# %%
# menghapus kolom yang tidak dibutuhkan untuk clustering
df_merge = df_merge.drop(columns=['Date', 'ProductID', 'Price', 'StoreID'])

# %%
df_merge.info()

# %%
df_merge.isna().sum()

# %% [markdown]
# ##### Membuat data baru untuk clustering

# %%
data_cluster_agg = df_merge.groupby('CustomerID').agg({'TransactionID': 'count', 'Qty': 'sum', 'TotalAmount': 'sum'}).reset_index()
data_cluster_agg = data_cluster_agg.rename(columns={'TransactionID': 'Total Transaction'})
data_cluster_agg

# %%
data_cluster_agg.info()

# %%
data_cluster_agg.isna().values.any()
# False = tidak ada missing value 

# %% [markdown]
# ##### K-Means Clustering

# %%
# Segmentasi berdasarkan data baru yang telah di agregat
# Membuat model clustering dengan K = 1 sampai K= 10 untuk
# menentukan K (banyaknya kelompok) yang optimal
X1 = data_cluster_agg[['Qty', 'TotalAmount']]
wcss = []
for n in range(1,11):
    cluster1 = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=100)
    cluster1.fit(X1)
    wcss.append(cluster1.inertia_)
print(wcss)

X2 = data_cluster_agg[['Total Transaction', 'TotalAmount']]
wcss2 = []
for n in range(1,11):
    cluster2 = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=100)
    cluster2.fit(X2)
    wcss2.append(cluster2.inertia_)
print(wcss2)

# %%
# Plot grafik WCSS vs. Nilai K = wcss pertama
plt.figure(figsize=(7,3))
plt.plot(list(range(1,11)), wcss, marker='o')
plt.title('WCSS1 vs. Amount of Cluster')
plt.xlabel('Amount of Cluster (K)')
plt.ylabel('WCSS')
plt.show()

# %%
# Plot grafik WCSS vs. Nilai K = wcss kedua
plt.figure(figsize=(7,3))
plt.plot(list(range(1,11)), wcss2, marker='o', color='red')
plt.title('WCSS2 vs. Amount of Cluster')
plt.xlabel('Amount of Cluster (K)')
plt.ylabel('WCSS')
plt.show()

# %% [markdown]
# mendapati bahwa K optimal dari kedua plot yaitu saat K = 3

# %% [markdown]
# ##### Membuat model clustering dengan K yang optimal

# %%
# cluster dari data X1
# membangun kembali model clustering dengan K = 3
# melatih model dengan data yang telah di agregasi
cluster1 = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=100)
cluster1.fit(X1)
labels1 = cluster1.labels_
centroids1 = cluster1.cluster_centers_

# %%
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
plt.scatter(x=data_cluster_agg['Qty'], y=data_cluster_agg['TotalAmount'], c=labels1, cmap='winter', s=50)
plt.scatter(x=centroids1[:,0], y=centroids1[:,1], c='red', s=200)
plt.title('Customer Segmentation based on Total Qty and Total Amount', fontsize=18)
plt.xlabel('Total Qty', fontsize=14)
plt.ylabel('Total Amount', fontsize=14)
plt.show()

# %%
# cluster dari data X2
# membangun kembali model clustering dengan K = 3
# melatih model dengan data yang telah di agregasi
cluster2 = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=100)
cluster2.fit(X2)
labels2 = cluster2.labels_
centroids2 = cluster2.cluster_centers_

# %%
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
plt.scatter(x=data_cluster_agg['Total Transaction'], y=data_cluster_agg['TotalAmount'], c=labels1, cmap='viridis', s=50)
plt.scatter(x=centroids2[:,0], y=centroids2[:,1], c='red', s=200)
plt.title('Customer Segmentation based on Total Transaction and Total Amount', fontsize=18)
plt.xlabel('Total Transaction', fontsize=14)
plt.ylabel('Total Amount', fontsize=14)
plt.show()

# %% [markdown]
# ````
# Dari kedua hasil model clustering di atas disimpulkan bahwa:
# 1. Segmentasi Pelanggan berdasarkan Total Amount dan Total Qty menghasilkan 3 segmentasi
# 2. Segmentasi Pelanggan berdasarkan Total Transaction dan Total Qty menghasilkan 3 segmentasi
# 


