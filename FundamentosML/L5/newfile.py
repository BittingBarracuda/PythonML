import numpy as np
from distances import euclidean

def get_centroid(data_points):
	return np.mean(data_points, axis = 1)

def recalculate_centroids(k, data_points, classes):
	centroids = np.zeros(shape = (k, ))
	for i in range(k):
		pos = np.where(classes == i)
		points = data_points[pos, :]
		centroids[i] = get_centroid(points)
	return centroids

def k_folds(k, data_points, centroids):
	keep = True
	while keep:
		distances =np.array( [[euclidean(point, centroid) for centroid in centroids] for point in data_points])
		new_classes = np.argmin(distances, axis = 1)
		if classes == new_classes:
			keep = False
		else:
			classes = new_classes
			centroids = recalculate_centroids(k, data_points, classes)
		return new_classes, centroids

centroids = np.array([[2, 10], [8, 4], [5, 8]])
data_points = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
centroids, classes = k_folds(3, data_points, centroids)


def get_clusters_dbscan(data_points, M, eps):
	clusters = []
	for point in data_points:
		distances = [euclidean(point, other) for other in data_points]
		close = np.where(distances < eps)
		if len(close)  - 1 >= M:
			aux_set = set(close)
			added = False
			for cluster in clusters:
				if aux_set & cluster != {}:
					cluster = cluster | aux_set
					added = True
					break
			if not added:
				clusters.append(aux_set)
	return clusters

M1, M2 = 2, 2
eps1, eps2 = np.sqrt(2), np.sqrt(10)
clusters1 = get_clusters_dbscan(data_points, M1, eps1)
clusters_2 = get_clusters_dbscan(data_points, M2, eps2)


	
	