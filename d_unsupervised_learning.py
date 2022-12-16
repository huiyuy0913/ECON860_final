import pandas
import matplotlib.pyplot as pyplot
import numpy
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

dataset = pandas.read_csv('results.csv')
# print(dataset)
dataset = dataset.values 
# print(dataset)

print('--------------------------------------------------------------------------------')
print('-------------------------------Solution for (d)---------------------------------')
print('--------------------------------------------------------------------------------')



def run_kmeans(n,dataset):
	print("Running kmeans with n=", n)

	machine = KMeans(n_clusters=n)  
	machine.fit(dataset)
	results = machine.predict(dataset)
	centroids = machine.cluster_centers_ 
	ssd = machine.inertia_ 
	if n>1:
		silhouette = silhouette_score(dataset, results, metric='euclidean')
	else:
		silhouette = 0
	return ssd, silhouette

result = [run_kmeans(i+1,dataset) for i in range(7)]
print(result)

ssd_result = [ i[0] for i in result]

pyplot.plot(range(1,8), ssd_result)
pyplot.savefig("kmeans_ssd.png")
pyplot.close()

silhouette_result = [ i[1] for i in result][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("kmeans_silhouette.png")
pyplot.close()

# print(silhouette_result.index(max(silhouette_result))+2)
print('In KMeans clustering, people will be divided into', silhouette_result.index(max(silhouette_result))+2, 'groups')









def run_kmedoids(n,dataset):
	print("Running kmedoids with n=", n)

	machine = KMedoids(n_clusters=n)  
	machine.fit(dataset)
	results = machine.predict(dataset)
	centroids = machine.cluster_centers_ 
	ssd = machine.inertia_ 
	if n>1:
		silhouette = silhouette_score(dataset, results, metric='euclidean')
	else:
		silhouette = 0
	return ssd, silhouette


result = [run_kmedoids(i+1,dataset) for i in range(7)]
print(result)

ssd_result = [ i[0] for i in result]

pyplot.plot(range(1,8), ssd_result)
pyplot.savefig("kmedoids_ssd.png")
pyplot.close()

silhouette_result = [ i[1] for i in result][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("kmedoids_silhouette.png")
pyplot.close()

# print(silhouette_result.index(max(silhouette_result))+2)
print('In KMedoids clustering, people will be divided into', silhouette_result.index(max(silhouette_result))+2, 'groups')







def run_gmm(n, dataset):   
	print("Running gmm with n=", n)
	machine = GaussianMixture(n_components=n)
	machine.fit(dataset)
	results = machine.predict(dataset)
	centroids = machine.means_
	silhouette = silhouette_score(dataset,results, metric = "euclidean")
	return silhouette

results = [run_gmm(i+2, dataset) for i in range(7)]
print(results)
pyplot.plot(range(2,9),results)
pyplot.savefig("gmm_silhouette.png")
pyplot.close()
# print(results.index(max(results))+2)
print('In Gaussian mixture model, people will be divided into', results.index(max(results))+2, 'groups')








