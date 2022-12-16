import pandas
from factor_analyzer import FactorAnalyzer
import numpy
numpy.set_printoptions(suppress=True) 
dataset = pandas.read_csv("dataset_final.csv")
print((dataset > 5).any())
# print(dataset)
dataset = dataset[(dataset != 0).all(1)]
print(dataset)
print('--------------------------------------------------------------------------------')
print('-------------------------------Solution for (c)---------------------------------')
print('--------------------------------------------------------------------------------')
machine = FactorAnalyzer(n_factors=40, rotation=None)
data = dataset.iloc[:,0:40]
# print(data)
machine.fit(data)
ev, v = machine.get_eigenvalues()
print(ev)

count = numpy.count_nonzero(ev > 1)
print("The number of personality traits from eigenvalues is " + str(count))






machine = FactorAnalyzer(n_factors=count-1, rotation=None)
machine.fit(data)
output = machine.loadings_
# print(output)

df = pandas.DataFrame(output)
df = df.abs()
maxValueIndex = df.idxmax(axis=1)
# print("Max values of row are at following columns :")
maxValueIndex = pandas.DataFrame(maxValueIndex)
# print(maxValueIndex)
print('--------------------------')
df2 = maxValueIndex.groupby(maxValueIndex.iloc[:, 0]).count()
print(df2)
# print('--------------------------')








machine = FactorAnalyzer(n_factors=count, rotation=None)
machine.fit(data)
output = machine.loadings_
# print(output)

df = pandas.DataFrame(output)
df = df.abs()
maxValueIndex = df.idxmax(axis=1)
# print("Max values of row are at following columns :")
maxValueIndex = pandas.DataFrame(maxValueIndex)
# print(maxValueIndex)
print('--------------------------')
df2 = maxValueIndex.groupby(maxValueIndex.iloc[:, 0]).count()
print(df2)
# print('--------------------------')







machine = FactorAnalyzer(n_factors=count+1, rotation=None)
machine.fit(data)
output = machine.loadings_
# print(output)

df = pandas.DataFrame(output)
df = df.abs()
maxValueIndex = df.idxmax(axis=1)
# print("Max values of row are at following columns :")
maxValueIndex = pandas.DataFrame(maxValueIndex)
# print(maxValueIndex)
print('--------------------------')
df2 = maxValueIndex.groupby(maxValueIndex.iloc[:, 0]).count()
print(df2)
print('--------------------------')
print('I will choose 6 as the number of personality traits since no question goes into the seventh factor when I increase factors to 7, so does the eighth factor.')






machine = FactorAnalyzer(n_factors=count-1, rotation='varimax')
machine.fit(data)
factor_loadings = machine.loadings_
# print(factor_loadings)

df = pandas.DataFrame(factor_loadings)
df = df.abs()
maxValueIndex = df.idxmax(axis=1)
# print("Max values of row are at following columns :")
maxValueIndex = pandas.DataFrame(maxValueIndex)
# print(maxValueIndex)
print('--------------------------')
df2 = maxValueIndex.groupby(maxValueIndex.iloc[:, 0]).count()
print(df2)
print('--------------------------')


data = data.values

print(data.shape)
print(factor_loadings.shape)

results = numpy.dot(data, factor_loadings) # multiply the datasets

# print(results)

pandas.DataFrame(results).round().to_csv("results.csv", index=False) 



