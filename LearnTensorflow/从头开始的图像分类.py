from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import load_model
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array

def delete_corrupted_images(training_folder):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(training_folder, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
    print("Deleted %d images" % num_skipped)


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
    train_img_path = r"D:\ROG\Pictures\PetImages"
    model_path = r'model/Imageclassificationfromscratch'
    cat_jpg = r'../images/test_set/cats/cat1.jpg'
    dog_jpg = r'../images/test_set/dogs/dog1.jpg'

    # delete_corrupted_images(train_img_path)
    # training(train_img_path, image_size, model_path)
    predict(dog_jpg, image_size, model_path)
