import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from dataCreator import *

N = 1200
a = 0.1
b = 0.2
tau = 15

nn = MLPRegressor(hidden_layer_sizes=100)
nn.n_iter_ = 200
nn.n_layers_ = 2
nn.n_outputs_ = 1

scaler = StandardScaler()

datasets = data_creator()  # массив с 100 датасетов
sheet_datasets = []
for dataset in datasets:
    sheet_datasets.append(create_dataset(dataset))


def scale_data(data):
    scaled_data = scaler.fit_transform(data.values.T)
    return pd.DataFrame(scaled_data).T


def unscale_data(scaled_data):
    unscaled_data = scaler.inverse_transform(scaled_data)
    return pd.DataFrame(unscaled_data)


def train():
    for sheet_dataset in sheet_datasets:
        scaled_dataset = scaler.fit_transform(sheet_dataset)
        scaled_dataset = pd.DataFrame(scaled_dataset)
        train_y = scaled_dataset[10]
        train_x = scaled_dataset.drop(10, axis=1)
        nn.fit(train_x, train_y)


def prediction(array):
    print(array)
    calculated = [0.546, 0.8313, 0.9034, 0.4575, 0.4562, 0.2642, 0.9076, 1.1753, 0.2816,
                  1.1399, 1.2056, 0.9617, 0.6693, 0.2189, 0.6081]
    # Нейросеть
    array_for_predict = array.drop('y_0', axis=1).values.reshape(1, -1)
    print(array_for_predict)
    print(array_for_predict[0])
    predicated = []
    for i in range(2001):
        predict = np.round(nn.predict(array_for_predict), 4)
        # print(predict)
        predicated.append(predict)
        del array_for_predict[0][0]
        # print(array_for_predict)
        array_for_predict.insert(9, i + 10, predict)  # (pd.DataFrame(predict))
        # print(array_for_predict)
    predicated = create_dataset(predicated)
    # Формула
    print(predicated)
    for i in range(tau - 1, 2000):
        calculated.append(
            round(calculated[i] - a * calculated[i] + b * calculated[i + 1 - tau] / (1 + calculated[i + 1 - tau] ** 10),
                  4))
    calculated = create_dataset(calculated)
    plt.plot(calculated._get_column_array(10)[100:])
    plt.plot(predicated._get_column_array(10)[100:])
    print(calculated._get_column_array(10))
    print(predicated._get_column_array(10))
    plt.xticks(np.linspace(0, 2000, 5))
    plt.axis([0, 2000, 0, 1.5])
    plt.show()
    delta = 0
    for i in range(calculated._get_column_array(10).size - 1):
        delta += abs(abs(calculated._get_column_array(10)[i]) - abs(predicated._get_column_array(10)[i]))
    print(delta)


test = [0.546, 0.8313, 0.9034, 0.4575, 0.4562, 0.2642, 0.9076, 1.1753, 0.2816,
        1.1399, 1.2056, 0.9617, 0.6693, 0.2189, 0.6081]
test = create_dataset(test)
print(test)
train()
for i in range(2000):
    predicted_value = prediction(test)
    calculated_value = round(test[i] - a * test[i] + b * test[i + 1 - tau] / (1 + test[i + 1 - tau] ** 10), 4)


# for row in scaled_dataset:
# scaled_train_x = scaler.fit_transform(train_x)
# nn.fit(scaled_dataset[row][:10].values.reshape(1, -1), np.array(scaled_dataset.iloc[row][10]))
# print(scaled_dataset[row][:10].reshape(1, -1))
# print(scaled_dataset.iloc[:][10])
