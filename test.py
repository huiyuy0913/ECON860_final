import pandas
from factor_analyzer import FactorAnalyzer
import numpy
from sklearn import linear_model
from sklearn import metrics

numpy.set_printoptions(suppress=True) 
dataset = pandas.read_csv("dataset_final.csv")
dataset = dataset.iloc[:,0:40]
dataset.replace(0, numpy.nan, inplace=True)
# dataset.dropna(inplace=True) 
dataset=dataset.dropna(how='all')
# print(dataset)

machine = FactorAnalyzer(n_factors=40, rotation=None)
data = dataset.iloc[:,0:40]
# print(data)
machine.fit(data)
ev, v = machine.get_eigenvalues()
# print(ev)

count = numpy.count_nonzero(ev > 1)
# print("The number of personality traits from eigenvalues is " + str(count))


machine = FactorAnalyzer(n_factors=count-1, rotation='varimax')
machine.fit(data)
output = machine.loadings_
# print(output)

df = pandas.DataFrame(output)
# print(df)
df = df.abs()
for i in range(6):
	locals()['df_'+str(i)] = pandas.DataFrame(df.iloc[:,i])
for i in range(6):
	locals()['df_'+str(i)]['rank_'+str(i)] = locals()['df_'+str(i)].rank()
	locals()['df_'+str(i)] = locals()['df_'+str(i)].sort_values(by=['rank_'+str(i)], ascending=False).reset_index()
	locals()['df_'+str(i)] = locals()['df_'+str(i)].rename(columns={"index": "index_"+str(i)})
df_total = pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5], axis=1)
print(df_total)










data = pandas.read_csv("results.csv")
math = pandas.read_csv("dataset_final.csv")
math = math[(math != 0).all(1)]
target = math.iloc[:,40].values 

data = data.values

machine = linear_model.LinearRegression()
machine.fit(data, target) 
slope = machine.coef_.tolist()
print(slope)
# slope = [ 1.40642495,0.06912387,-2.63890123,1.45591775,1.11590871,-0.14888214]
df_new_total = df_total
for idx, v in enumerate(slope ):
	df_new_total[idx] = df_new_total[idx].apply(lambda x: x*v)
    # print(idx, x)
print(df_new_total)
df_combine = pandas.concat([df_new_total.iloc[:, 0:3].rename(columns={"index_0": "index", 0: "load","rank_0": "rank"})
	, df_new_total.iloc[:, 3:6].rename(columns={"index_1": "index", 1: "load","rank_1": "rank"})
	, df_new_total.iloc[:, 6:9].rename(columns={"index_2": "index", 2: "load","rank_2": "rank"})
	, df_new_total.iloc[:, 9:12].rename(columns={"index_3": "index", 3: "load","rank_3": "rank"})
	, df_new_total.iloc[:, 12:15].rename(columns={"index_4": "index", 4: "load","rank_4": "rank"})
	, df_new_total.iloc[:, 15:18].rename(columns={"index_5": "index", 5: "load","rank_5": "rank"})]).abs().sort_values(by='load', ascending=False).reset_index(drop=True)
print(df_combine.head(30))
print(df_combine['index'].head(30).unique())
top = df_combine['index'].head(30).unique()
print(top[0:20])


df_final = pandas.concat([df_total['index_0'].head(4)
	, df_total['index_1'].head(4)
	, df_total['index_2'].head(4)
	, df_total['index_3'].head(4)
	, df_total['index_4'].head(4)
	, df_total['index_5'].head(4)]).reset_index(drop=True)
print(df_final)
print(df_final.nunique())






