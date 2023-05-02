import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('anomaly_data.csv')

fig = plt.figure(figsize=(10, 5))
plt.scatter(data.loc[:, 'X1'], data.loc[:, 'X2'])
plt.title('data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

x1 = data.loc[:, 'X1']
x2 = data.loc[:, 'X2']

fig1 = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(x1, bins=100)
plt.title('X1 distribution')
plt.xlabel('X1')
plt.ylabel('counts')
plt.subplot(122)
plt.hist(x2, bins=100)
plt.title('X2 distribution')
plt.xlabel('X2')
plt.ylabel('counts')
plt.show()

x1_mean = x1.mean()
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()

x1_range = np.linspace(0, 20, 300)
x1_normal = norm.pdf(x1_range, x1_mean, x1_sigma)
x2_range = np.linspace(0, 20, 300)
x2_normal = norm.pdf(x2_range, x2_mean, x2_sigma)

fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x1_range, x1_normal)
plt.title('normal p(X1)')
plt.subplot(122)
plt.plot(x2_range, x2_normal)
plt.title('normal p(X2)')
plt.show()

ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(data)

y_predict = ad_model.predict(data)

fig4 = plt.figure(figsize=(10, 5))
orginal_data = plt.scatter(data.loc[:, 'X1'], data.loc[:, 'X2'], marker='x')
anomaly_data = plt.scatter(data.loc[:, 'X1'][y_predict == -1], data.loc[:, 'X2'][y_predict == -1], marker='o',
                           facecolor='none', edgecolors='red')
plt.title('anomaly detection result')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend((orginal_data, anomaly_data), ('orginal_data', 'anomaly_data'))
plt.show()
