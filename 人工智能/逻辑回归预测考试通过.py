import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('examdata.csv')
# print(data.head())

# fig1 = plt.figure()
# plt.scatter(data.loc[:, 'Exam1'], data.loc[:, 'Exam2'])
# plt.title('Exam1-Exam2')
# plt.xlabel('Exam1')
# plt.ylabel('Exam2')
# plt.show()

mask = data.loc[:, 'Pass'] == 1

# fig2 = plt.figure()
# passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
# failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
# plt.title('Exam1-Exam2')
# plt.xlabel('Exam1')
# plt.ylabel('Exam2')
# plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()

x = data.drop(['Pass'], axis=1)
y = data.loc[:, 'Pass']
x1 = data.loc[:, 'Exam1']
x2 = data.loc[:, 'Exam2']

LR = LogisticRegression()
LR.fit(x, y)

y_predict = LR.predict(x)
print(y_predict)

accuracy = accuracy_score(y, y_predict)
print(accuracy)

y_test = LR.predict([[70, 65]])
print('passed' if y_test else 'failed')

theta0 = LR.intercept_
theta1, theta2 = LR.coef_[0][0], LR.coef_[0][1]

X2_new = -(theta0 + theta1 * x1) / theta2
# fig3 = plt.figure()
# passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
# failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
# plt.plot(x1, X2_new)
# plt.title('Exam1-Exam2')
# plt.xlabel('Exam1')
# plt.ylabel('Exam2')
# plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()

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

fig4 = plt.figure()
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
plt.plot(X1_new, X2_new_boundary)
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()

plt.plot(X1_new, X2_new_boundary)
plt.show()
