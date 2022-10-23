import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from dataCreator import *

N = 1200
a = 0.1
b = 0.2
tau = 15

nn = MLPRegressor(hidden_layer_sizes=50)  # Configuration
nn.n_layers_ = 2
nn.n_outputs_ = 1

scaler = StandardScaler()


def train():  # Train function
    start = time.time()
    for i in range(99):
        dataset = pd.read_csv(f"datasets/dataset{i}").drop('Unnamed: 0', axis=1)
        train_y = dataset['y_0']
        train_x = dataset.drop('y_0', axis=1)
        scaled_train_x = pd.DataFrame(scaler.fit_transform(train_x))
        if (i % 5 != 0) | (i == 0):
            nn.fit(scaled_train_x, train_y)
            # print(f"Training set score: {nn.score(scaled_train_x, train_y):.3%}")
        else:
            print(f"Testing set score: {nn.score(scaled_train_x, train_y):.3%}\n")
    finish = time.time()
    print("Train finished!")
    return finish - start  # Train time


def get_results(dataset):  # Analysis on dataset
    calculated = dataset['x_0']
    predicted = np.round(nn.predict(scaler.fit_transform(dataset)), 4)

    # Statistics
    print("Elements in hidden layer:", nn.hidden_layer_sizes)
    print("MSE:", mean_squared_error(calculated[15:], predicted[:len(predicted) - tau]))
    print("MAE:", mean_absolute_error(calculated[15:], predicted[:len(predicted) - tau]))

    # Plot
    plt.plot(calculated[100:], 'r-', label="Calculated")
    plt.plot([i for i in range(100 + tau, len(predicted) + tau)], predicted[100:], 'b-', label="Predicted")
    plt.xticks(np.linspace(100, 2100, 5))
    plt.axis([100, 2100, 0, 2])
    plt.legend()
    plt.show()


def test_nn(n_datasets):  # Analysis NN
    mse, mae, m_score = 0, 0, 0
    for i in range(n_datasets - 1):
        dataset = create_sheet(create_data()).drop('y_0', axis=1)
        calculated = dataset['x_0']
        predicted = np.round(nn.predict(scaler.fit_transform(dataset)), 4)
        mse += mean_squared_error(calculated[15:], predicted[:len(predicted) - tau])
        mae += mean_absolute_error(calculated[15:], predicted[:len(predicted) - tau])

    print(f"Elements in hidden layer: {nn.hidden_layer_sizes}")
    print(f"MSE(for {n_datasets}): {mse / n_datasets}")
    print(f"MAE(for {n_datasets}): {mae / n_datasets}")
    print(f"Train time is: {train_time}sec.")


train_time = train()

# new_test = create_sheet(create_data()).drop('y_0', axis=1)  # NN test example
# get_results(new_test)

test_nn(100)
