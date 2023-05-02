import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift, estimate_bandwidth

data = pd.read_csv('data.csv')

x = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']

pd.value_counts(y)

# fig1 = plt.figure()
# label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
# plt.title('labled data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.show()

KM = KMeans(n_clusters=3, random_state=0)
KM.fit(x)

centers = KM.cluster_centers_
# fig3 = plt.figure()
# label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
# plt.title('labled data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
# plt.show()

y_predict_test = KM.predict([[80, 60]])

y_predict = KM.predict(x)

accuracy = accuracy_score(y, y_predict)
# fig1 = plt.subplot(121)
# label0 = plt.scatter(x.loc[:, 'V1'][y_predict == 0], x.loc[:, 'V2'][y_predict == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y_predict == 1], x.loc[:, 'V2'][y_predict == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y_predict == 2], x.loc[:, 'V2'][y_predict == 2])
# plt.title('predicted data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
#
# fig1 = plt.subplot(122)
# label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
# plt.title('labled data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
# plt.show()

y_corrected = []
for i in y_predict:
    if i == 0:
        y_corrected.append(2)
    elif i == 1:
        y_corrected.append(0)
    else:
        y_corrected.append(1)

# print(accuracy_score(y, y_corrected))

y_corrected = np.array(y_corrected)
# fig6 = plt.subplot(121)
# label0 = plt.scatter(x.loc[:, 'V1'][y_corrected == 0], x.loc[:, 'V2'][y_corrected == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y_corrected == 1], x.loc[:, 'V2'][y_corrected == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y_corrected == 2], x.loc[:, 'V2'][y_corrected == 2])
# plt.title('corrected data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
#
# fig7 = plt.subplot(122)
# label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
# plt.title('labled data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
# plt.show()

# KNN model
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x, y)

y_predict_knn_test = KNN.predict([[80, 60]])
y_predict_knn = KNN.predict(x)
# print(y_predict_knn_test)
# print('knn accuracy:', accuracy_score(y, y_predict_knn))
#
# fig6 = plt.subplot(121)
# label0 = plt.scatter(x.loc[:, 'V1'][y_predict_knn == 0], x.loc[:, 'V2'][y_predict_knn == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y_predict_knn == 1], x.loc[:, 'V2'][y_predict_knn == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y_predict_knn == 2], x.loc[:, 'V2'][y_predict_knn == 2])
# plt.title('knn results')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
#
# fig7 = plt.subplot(122)
# label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
# label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
# label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
# plt.title('labled data')
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.scatter(centers[:, 0], centers[:, 1])
# plt.show()

# meanshift
bw = estimate_bandwidth(x, n_samples=5)
# print(bw)

ms = MeanShift(bandwidth=20)
ms.fit(x)

y_predict_ms = ms.predict(x)
fig6 = plt.subplot(121)
label0 = plt.scatter(x.loc[:, 'V1'][y_predict_ms == 0], x.loc[:, 'V2'][y_predict_ms == 0])
label1 = plt.scatter(x.loc[:, 'V1'][y_predict_ms == 1], x.loc[:, 'V2'][y_predict_ms == 1])
label2 = plt.scatter(x.loc[:, 'V1'][y_predict_ms == 2], x.loc[:, 'V2'][y_predict_ms == 2])
plt.title('ms results')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.scatter(centers[:, 0], centers[:, 1])

fig7 = plt.subplot(122)
label0 = plt.scatter(x.loc[:, 'V1'][y == 0], x.loc[:, 'V2'][y == 0])
label1 = plt.scatter(x.loc[:, 'V1'][y == 1], x.loc[:, 'V2'][y == 1])
label2 = plt.scatter(x.loc[:, 'V1'][y == 2], x.loc[:, 'V2'][y == 2])
plt.title('labled data')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()