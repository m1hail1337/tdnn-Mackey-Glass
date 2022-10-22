import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataCreator import dataCreator


def create_dataset(array, length=10, skip_rows=0, delay=4, y_size=1) -> pd.DataFrame:
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


#data1 = dataCreator()[0]
#plt.plot(data1)
#plt.show()
#print(data1)

