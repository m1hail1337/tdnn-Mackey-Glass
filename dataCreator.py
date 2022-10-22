import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 1200
a = 0.1
b = 0.2
tau = 15


def create_dataset(array, length=10, skip_rows=0, delay=4, y_size=1) -> pd.DataFrame:  # data to pandas sheet
    columns = np.append([f"x_{j}" for j in range(length)], [f"y_{k}" for k in range(y_size)])
    result = []

    for i in range(skip_rows, len(array) - length - y_size - delay + 1):
        x_index_start = i
        x_index_end = i + length
        y_index_start = i + length + delay
        y_index_end = i + length + delay + y_size

        row = np.append(array[x_index_start:x_index_end], array[y_index_start:y_index_end])
        result.append(row)
    result = pd.DataFrame(result, columns=columns, dtype="float")
    return result


def data_creator():
    datasets = []
    for i in range(10):
        dataset = []
        for j in range(tau):
            dataset.append(round(random.uniform(0.2, 1.3), 4))
        for k in range(tau - 1, 2000):
            dataset.append(
                round(dataset[k] - a * dataset[k] + b * dataset[k + 1 - tau] /
                      (1 + dataset[k + 1 - tau] ** 10), 4))
        datasets.append(dataset)
    return datasets


#for dataset in data_creator():
#    print(dataset, "\n")
#data1 = data_creator()[0]._get_column_array(0)[100:]
#plt.plot(data1)
#plt.xticks(np.linspace(0, 2000, 5))
#plt.axis([0, 2000, 0, 1.4])
#plt.show()

