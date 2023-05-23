import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN


def extract_data(data, time_step):
    X = []
    Y = []
    for i in range(len(data) - time_step):
        X.append([a for a in data[i:i + time_step]])
        Y.append(data[i + time_step])
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, np.array(Y)


def training():
    model = Sequential()
    # add RNN layer
    model.add(SimpleRNN(units=5, input_shape=(time_step, 1), activation='relu'))
    # add output layer
    model.add(Dense(units=1, activation='linear'))
    # configure the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # train the model
    model.fit(X, Y, batch_size=30, epochs=100)
    # save the model
    model.save(model_path)


def prediction():
    model = load_model(model_path)
    # y_train_predict = model.predict(X) * max(price)
    # y_train = [i * max(price) for i in Y]
    # plt.figure(figsize=(8, 5))
    # plt.plot(y_train, label='real price')
    # plt.plot(y_train_predict, label='predict price')
    # plt.title('close price')
    # plt.xlabel('time')
    # plt.ylabel('price')
    # plt.legend()
    # plt.show()

    data_test = pd.read_csv('data/中船科技600072test.csv')
    price_test = data_test.loc[:, '收盘']
    price_test_norm = price_test / max(price)
    x_test_norm, y_test_norm = extract_data(price_test_norm, time_step)
    y_test_predict = model.predict(x_test_norm) * max(price)
    y_test = [i * max(price) for i in y_test_norm]
    # plt.figure(figsize=(8, 5))
    # plt.plot(y_test, label='real price')
    # plt.plot(y_test_predict, label='predict price')
    # plt.title('close price')
    # plt.xlabel('time')
    # plt.ylabel('price')
    # plt.legend()
    # plt.show()

    result_y_test = np.array(y_test).reshape(-1, 1)
    result_y_test_predict = y_test_predict
    print(result_y_test.shape, result_y_test_predict.shape)
    result = np.concatenate((result_y_test, result_y_test_predict), axis=1)
    print(result.shape)
    result = pd.DataFrame(result, columns=['real_price_test', 'predict_price_test'])
    result.to_csv('data/中船科技600072predict.csv')


if __name__ == '__main__':
    data_path = 'data/中船科技600072.csv'
    model_path = 'model/RNN股价预测'

    data = pd.read_csv(data_path)
    # print(data.head())

    price = data.loc[:, '收盘']
    # print(price.head())

    # 归一化处理
    price_norm = price / max(price)
    # print(price_norm)

    # plt.figure(figsize=(8, 5))
    # plt.plot(price)
    # plt.title('close price')
    # plt.xlabel('time')
    # plt.ylabel('price')
    # plt.show()

    time_step = 8

    X, Y = extract_data(price_norm, time_step)

    # training()
    prediction()
