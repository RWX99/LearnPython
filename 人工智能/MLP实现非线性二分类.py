from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("MLP_test_data.csv")

x = data.drop(['y'], axis=1)
y = data.loc[:, 'y']

# fig1 = plt.figure(figsize=(5, 5))
# passed = plt.scatter(x.loc[:, 'X1'][y == 1], x.loc[:, 'X2'][y == 1])
# failed = plt.scatter(x.loc[:, 'X1'][y == 0], x.loc[:, 'X2'][y == 0])
# plt.legend((passed, failed), ('passed', 'failed'))
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('raw data')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

mlp = Sequential()
mlp.add(Dense(units=20, activation='sigmoid', input_dim=2))
mlp.add(Dense(units=1, activation='sigmoid'))

print(mlp.summary())

mlp.compile(loss='binary_crossentropy', optimizer='adam')

mlp.fit(x_train, y_train, epochs=3000)

y_train_predict = mlp.predict_classes(x_train)
accuracy_train = accuracy_score(y_train, y_train_predict)
y_test_predict = mlp.predict_classes(x_test)
accuracy_test = accuracy_score(y_test, y_test_predict)

y_train_predict_form = pd.Series(i[0] for i in y_train_predict)

xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
x_range = np.c_[xx.ravel(), yy.ravel()]
y_range_predict = mlp.predict_classes(x_range)

y_range_predict_form = pd.Series(i[0] for i in y_range_predict)

fig2 = plt.figure(figsize=(5, 5))
passed_predict = plt.scatter(x_range[:, 0][y_range_predict_form == 1], x_range[:, 1][y_range_predict_form == 1])
failed_predict = plt.scatter(x_range[:, 0][y_range_predict_form == 0], x_range[:, 1][y_range_predict_form == 0])
passed = plt.scatter(x.loc[:, 'X1'][y == 1], x.loc[:, 'X2'][y == 1])
failed = plt.scatter(x.loc[:, 'X1'][y == 0], x.loc[:, 'X2'][y == 0])
plt.legend((passed, failed, passed_predict, failed_predict), ('passed', 'failed', 'passed_predict', 'failed_predict'))
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('prediction result')
plt.show()
