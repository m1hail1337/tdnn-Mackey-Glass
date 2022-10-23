import matplotlib.pyplot as plt
import numpy as np
x = [10, 30, 50, 70, 100, 200]  # Hidden layer sizes
mse_y = [0.017759, 0.006964, 0.005387, 0.004919, 0.004823, 0.003677]    # Type tour data in this lists to
mae_y = [0.067182, 0.052699, 0.046580, 0.044452, 0.053431, 0.041812]    # create a plot with metrics
time_y = [13.6716, 13.4926, 22.2222, 21.8459, 24.6636, 25.9793]
plt.plot(x, time_y, 'r-')
plt.plot(x, time_y, "p")
plt.xticks(np.linspace(0, 200, 5))
plt.axis([0, 200, 0, 30])
plt.xlabel("Hidden layer size")
plt.ylabel("Training time")     # or MSE or MAE
plt.show()
