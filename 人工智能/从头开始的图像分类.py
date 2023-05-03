from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.utils import load_img, img_to_array
from keras.models import load_model
from matplotlib import pyplot as plt
import os


def training(training_folder, image_size, model_path):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory(training_folder, target_size=image_size, batch_size=32,
                                                     class_mode='binary')

    model = Sequential()
    # 卷积层
    model.add(Conv2D(32, (3, 3), input_shape=(*image_size, 3), activation='relu'))
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

    model.fit(training_set, epochs=1)

    model.save(model_path)


def predict(img_path, image_size, model_path):
    # Run inference on new data
    plt.figure()
    img = load_img(img_path, target_size=image_size)
    img = img_to_array(img)
    img = img.astype('float32') / 255
    img = img.reshape(1, *image_size, 3)
    result = load_model(model_path).predict(img)[0][0]
    plt.imshow(load_img(img_path))
    plt.title(f"{100 * (1 - result):.2f}% is cat and {100 * result:.2f}% is dog.")
    plt.show()


if __name__ == '__main__':
    # 训练数据下载地址
    # https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
    image_size = (180, 180)
    model_path = r'model/Imageclassificationfromscratch'
    # training(r"D:\软件\PetImages", image_size, model_path)
    cat_jpg = r'C:\Users\liangty\Desktop\1.jpg'
    dog_jpg = r'C:\Users\liangty\Desktop\2.jpg'
    predict(cat_jpg, image_size, model_path)
