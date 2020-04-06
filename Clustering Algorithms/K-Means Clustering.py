#K-Means Clustering

#importing libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#importing the datasets
data=pd.read_csv("Mall_Customers.csv")
x=data.iloc[:,3:].values

#selecting no.of clusters using elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")  
plt.xlabel("No.of Clusters")
plt.ylabel("WCSS")
plt.show()  

#creating K-Means cluster
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=0)
y_means=kmeans.fit_predict(x)

#Visualizing the Clusters
plt.scatter(x[y_means==0,0],x[y_means==0,1],c="red",s=100,label="Cluster 1")
plt.scatter(x[y_means==1,0],x[y_means==1,1],c="blue",s=100,label="Cluster 2")
plt.scatter(x[y_means==2,0],x[y_means==2,1],c="green",s=100,label="Cluster 3")
plt.scatter(x[y_means==3,0],x[y_means==3,1],c="yellow",s=100,label="Cluster 4")
plt.scatter(x[y_means==4,0],x[y_means==4,1],c="black",s=100,label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="cyan",s=100,label="centroids")
plt.title("Clusters of Clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()