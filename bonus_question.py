import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset_results = pandas.read_csv("results.csv")
dataset = pandas.read_csv("dataset_final.csv")
dataset = dataset[(dataset != 0).all(1)]
# print("Dataset")
# print(dataset)
# print(math)
target = dataset.iloc[:,40].values 
# print('Target')
# print(target)
data = dataset_results.values
# print('Data')
# print(data)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

machine = linear_model.LinearRegression()
machine.fit(data_train, target_train) # train the machine
r_sq = machine.score(data_train, target_train)
prediction = machine.predict(data_test)
# print(prediction) #y_hat
# print(len(prediction))
print("R2 score for linear regression:")
print(metrics.r2_score(target_test, prediction))


data = dataset.iloc[:, 0:40].values 

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)
machine = linear_model.LinearRegression()
machine.fit(data_train, target_train) # train the machine
r_sq = machine.score(data_train, target_train)
prediction = machine.predict(data_test)
# print(prediction) #y_hat
# print(len(prediction))
print("R2 score for linear regression:")
print(metrics.r2_score(target_test, prediction))
