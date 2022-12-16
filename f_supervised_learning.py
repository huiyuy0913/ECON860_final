import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pandas.read_csv("results.csv")
math = pandas.read_csv("dataset_final.csv")
math = math[(math != 0).all(1)]
# print("Dataset")
# print(dataset)
# print(math)
target = math.iloc[:,40].values 
# print('Target')
# print(target)
data = dataset.values
# print('Data')
# print(data)
print('--------------------------------------------------------------------------------')
print('-------------------------------Solution for (f)---------------------------------')
print('--------------------------------------------------------------------------------')


data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# print('Data_test')
# print(len(data_test))
# print('Target_test')
# print(len(target_test))

# print('Data_train')
# print(len(data_train))
# print('Target_train')
# print(len(target_train))


machine = linear_model.LinearRegression()
machine.fit(data_train, target_train) # train the machine
r_sq = machine.score(data_train, target_train)
# print(f"coefficient of determination: {r_sq}")
# print(f"intercept: {machine.intercept_}")
# print(f"slope: {machine.coef_}")
prediction = machine.predict(data_test)
# print(prediction) #y_hat
# print(len(prediction))
print("R2 score for linear regression:")
print(metrics.r2_score(target_test, prediction))



data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2) 



machine = linear_model.LogisticRegression(solver='lbfgs', max_iter=100) # when max_iter=10000, R^2 is still less than that in linear regression
machine.fit(data_train, target_train) # train the machine

prediction = machine.predict(data_test)
# print(prediction) #y_hat
# print(len(prediction))

print("R2 score for linear regression:") 
print(metrics.r2_score(target_test, prediction))





