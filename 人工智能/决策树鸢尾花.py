import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

data = pd.read_csv('data/iris.csv')

x = data.drop(['label'], axis=1)
y = data.loc[:, 'label']
# print(x.shape)
# print(y.shape)

dc_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)
dc_tree.fit(x, y)

y_predict = dc_tree.predict(x)
accuracy = accuracy_score(y, y_predict)
# print(accuracy)

fig = plt.figure(figsize=(10, 10))
tree.plot_tree(dc_tree, filled='True',feature_names=['SepalLength','SepalWidth','PetalLength','PetalWidth'],class_names=['setosa','versicolor','virginica'])

dc_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1)
dc_tree.fit(x, y)
fig = plt.figure(figsize=(20, 20))
tree.plot_tree(dc_tree, filled='True',feature_names=['SepalLength','SepalWidth','PetalLength','PetalWidth'],class_names=['setosa','versicolor','virginica'])
