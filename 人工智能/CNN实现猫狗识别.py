from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt
from matplotlib.image import imread

train_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('../images/training_set', target_size=(50, 50), batch_size=32,
                                                 class_mode='binary')

model = Sequential()
# 卷积层
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# 卷积层
model.add(Conv2D(32, (3, 3), activation='relu'))
# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# flattening layer
model.add(Flatten())
# FC layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print(model.summary())

model.fit_generator(training_set, epochs=25)

accuracy = model.evaluate_generator(training_set)

test_set = train_datagen.flow_from_directory('../images/test_set', target_size=(50, 50), batch_size=32,
                                             class_mode='binary')
accuracy_test = model.evaluate_generator(test_set)

pic_dog = r"D:\桌面\dog.jpg"
pic_dog = load_img(pic_dog, target_size=(50, 50))
pic_dog = img_to_array(pic_dog)
pic_dog /= 255
pic_dog = pic_dog.reshape(1, 50, 50, 3)
result = np.argmax(model.predict(pic_dog), axis=1)
print(result)

pic_cat = r"D:\桌面\cat.jpg"
pic_cat = load_img(pic_cat, target_size=(50, 50))
pic_cat = img_to_array(pic_cat)
pic_cat /= 255
pic_cat = pic_cat.reshape(1, 50, 50, 3)
result = np.argmax(model.predict(pic_cat), axis=1)
print(result)

print(training_set.class_indices)

font2 = {'family': 'SimHei', 'weight': 'normal', 'size': 20}
mlp.rcParams['font.family'] = 'SimHei'
mlp.rcParams['axes.unicode_minus'] = False
a = [i for i in range(1, 10)]
fig = plt.figure(figsize=(10, 10))
for i in a:
    img_name = str(i) + '.jpg'
    img_ori = load_img(img_name, target_size=(50, 50))
    img = img_to_array(img_ori)
    img = img.astype('float32') / 255
    img = img.reshape(1, 50, 50, 3)
    result = np.argmax(model.predict(img), axis=1)
    img_ori = load_img(img_name, target_size=(250, 250))
    plt.subplot(3, 3, i)
    plt.imshow(img_ori)
    plt.title('预测为：狗狗' if result[0] == 0 else '预测为：猫咪')
plt.show()
