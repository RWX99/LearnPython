import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('chip_test.csv')
# print(data.head())

mask = data.loc[:, 'pass'] == 1

# fig1 = plt.figure()
# passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
# failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
# plt.title('test1-test2')
# plt.xlabel('test1')
# plt.ylabel('test2')
# plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()

x = data.drop(['pass'], axis=1)
y = data.loc[:, 'pass']
x1 = data.loc[:, 'test1']
x2 = data.loc[:, 'test2']

X1_2 = x1 * x1
X2_2 = x2 * x2
X1_X2 = x1 * x2
X_new = {'X1': x1, 'X2': x2, 'X1_2': X1_2, 'X2_2': X2_2, 'X1_X2': X1_X2}
X_new = pd.DataFrame(X_new)

LR2 = LogisticRegression()
LR2.fit(X_new, y)

y2_predict = LR2.predict(X_new)
accuracy2 = accuracy_score(y, y2_predict)

X1_new = x1.sort_values()
theta0 = LR2.intercept_
theta1, theta2, theta3 = LR2.coef_[0][0], LR2.coef_[0][1], LR2.coef_[0][2]
theta4, theta5 = LR2.coef_[0][3], LR2.coef_[0][4]
a = theta4
b = theta5 * X1_new + theta2
c = theta0 + theta1 * X1_new + theta3 * X1_new * X1_new
X2_new_boundary = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


# fig4 = plt.figure()
# passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
# failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
# plt.plot(X1_new, X2_new_boundary)
# plt.title('test1-test2')
# plt.xlabel('test1')
# plt.ylabel('test2')
# plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()


def f(x):
    a = theta4
    b = theta5 * x + theta2
    c = theta0 + theta1 * x + theta3 * x * x
    X2_new_boundary1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    X2_new_boundary2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return X2_new_boundary1, X2_new_boundary2


# X2_new_boundary1 = []
# X2_new_boundary2 = []
# for x in X1_new:
#     X2_new_boundary1.append(f(x)[0])
#     X2_new_boundary2.append(f(x)[1])
# print(X2_new_boundary1, X2_new_boundary2)

# fig4 = plt.figure()
# passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
# failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
# plt.plot(X1_new, X2_new_boundary1)
# plt.plot(X1_new, X2_new_boundary2)
# plt.title('test1-test2')
# plt.xlabel('test1')
# plt.ylabel('test2')
# plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()

X1_range = [-0.9 + x / 10000 for x in range(0, 19000)]
X1_range = np.array(X1_range)
X2_new_boundary1 = []
X2_new_boundary2 = []
for x in X1_range:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])

fig4 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
plt.plot(X1_range, X2_new_boundary1)
plt.plot(X1_range, X2_new_boundary2)
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
