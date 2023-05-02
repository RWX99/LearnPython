import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

data_train = pd.read_csv('T-R-train.csv')
x_train = data_train.loc[:, 'T']
y_train = data_train.loc[:, 'rate']

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(x_train, y_train)
# plt.title('raw data')
# plt.xlabel('temperature')
# plt.ylabel('rate')
# plt.show()

x_train = np.array(x_train).reshape(-1, 1)
lr1 = LinearRegression()
lr1.fit(x_train, y_train)

data_test = pd.read_csv('T-R-test.csv')
x_test = data_test.loc[:, 'T']
y_test = data_test.loc[:, 'rate']

x_test = np.array(x_test).reshape(-1, 1)

y_train_predict = lr1.predict(x_train)
y_test_predict = lr1.predict(x_test)

r2_train = r2_score(y_train, y_train_predict)
r2_test = r2_score(y_test, y_test_predict)

x_range = np.linspace(40, 90, 300).reshape(-1, 1)
y_range_predict = lr1.predict(x_range)

# fig1 = plt.figure(figsize=(5, 5))
# plt.plot(x_range, y_range_predict)
# plt.scatter(x_train, y_train)
# plt.title('prediction data')
# plt.xlabel('temperature')
# plt.ylabel('rate')
# plt.show()

poly2 = PolynomialFeatures(degree=2)
x_2_train = poly2.fit_transform(x_train)
x_2_test = poly2.transform(x_test)

lr2 = LinearRegression()
lr2.fit(x_2_train, y_train)
y_2_train_predict = lr2.predict(x_2_train)
y_2_test_predict = lr2.predict(x_2_test)
r2_2_train = r2_score(y_train, y_2_train_predict)
r2_2_test = r2_score(y_test, y_2_test_predict)

x_2_range = np.linspace(40, 90, 300).reshape(-1, 1)
x_2_range = poly2.transform(x_2_range)
y_2_range_predict = lr2.predict(x_2_range)

# fig2 = plt.figure(figsize=(5, 5))
# plt.plot(x_range, y_2_range_predict)
# plt.scatter(x_train, y_train)
# plt.scatter(x_test, y_test)
# plt.title('polynomial prediction result (2)')
# plt.xlabel('temperature')
# plt.ylabel('rate')
# plt.show()

poly5 = PolynomialFeatures(degree=5)
x_5_train = poly5.fit_transform(x_train)
x_5_test = poly5.transform(x_test)

lr5 = LinearRegression()
lr5.fit(x_5_train, y_train)
y_5_train_predict = lr5.predict(x_5_train)
y_5_test_predict = lr5.predict(x_5_test)
r2_5_train = r2_score(y_train, y_5_train_predict)
r2_5_test = r2_score(y_test, y_5_test_predict)

x_5_range = np.linspace(40, 90, 300).reshape(-1, 1)
x_5_range = poly5.transform(x_5_range)
y_5_range_predict = lr5.predict(x_5_range)

fig3 = plt.figure(figsize=(5, 5))
plt.plot(x_range, y_5_range_predict)
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.title('polynomial prediction result (5)')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()