from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

img_path = r"D:\桌面\cat.jpg"
img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)

model_vgg = VGG16(weights='imagenet', include_top=False)
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
# print(x.shape)

# 特征提取
features = model_vgg.predict(x)
# print(features.shape)
features = features.reshape(1, 7 * 7 * 512)
# print(features.shape)

fig = plt.figure(figsize=(5, 5))
img = load_img(img_path, target_size=(224, 224))
plt.imshow(img)


def modelProcess(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    x_vgg = model.predict(x)
    x_vgg = x_vgg.reshape(1, 25088)
    return x_vgg


model_vgg = VGG16(weights='imagenet', include_top=False)
folder = r'../images/training_set/cats'
dirs = os.listdir(folder)
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == '.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]

features1 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    features_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features1[i] = features_i

folder = r'../images/training_set/dogs'
dirs = os.listdir(folder)
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == '.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]

features2 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    features_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features2[i] = features_i

print(features1.shape, features2.shape)
y1 = np.zeros(300)
y2 = np.ones(300)

x = np.concatenate((features1, features2), axis=0)
y = np.concatenate((y1, y2), axis=0)
y = y.reshape(-1, 1)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
print(x_train.shape, x_test.shape, x.shape)

model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=25088))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50)

y_train_predict = np.argmax(model.predict(x_train), axis=1)
accuracy_train = accuracy_score(y_train, y_train_predict)
print(accuracy_train)
