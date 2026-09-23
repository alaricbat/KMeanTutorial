import numpy as np

class CustomKMeans:
    def __init__(self, n_clusters=5):
        self.k = n_clusters
        self.labels = []

    def fit(self, X):
        self.X_train = np.array(X)
        self.__kmeans_init_centroid()
        while True:
            self.labels = self.__kmeans_assign_labels(self.X_train)
            new_centroids = self.__kmeans_update_centroid()
            if self.__kmeans_has_converged(new_centroids=new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        return self.__kmeans_assign_labels(X)

    def __kmeans_init_centroid(self):
        self.centroids = self.X_train[np.random.choice(a=self.X_train.shape[0], size=self.k, replace=False)]

    def __kmeans_assign_labels(self, X):
        distances = np.sum((self.X_train[:, np.newaxis, :] - self.centroids)**2, axis=2)
        return np.argmin(distances, axis=1)

    def __kmeans_update_centroid(self):
        centroids = np.zeros((self.k, self.X_train.shape[1]))
        for i in range(self.k):
            Xi = self.X_train[self.labels == i, :]
            if len(Xi) == 0:
                centroids[i, :] = self.centroids[i, :]
            else:
                centroids[i, :] = np.mean(Xi, axis=0)
        return centroids


    def __kmeans_has_converged(self, new_centroids) -> bool:
        return np.allclose(self.centroids, new_centroids, atol=1e-4)