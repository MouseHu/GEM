import numpy as np

# 3M
Q = [0.235, 0.201, 0.565, 0.276, 0.273]
err = [0.011, 0.010, 0.010, 0.010, 0.028]
print(np.mean(Q))
print(np.mean(err))

# 5M
Q = [0.295, 0.211, 0.297, 0.378, 0.333]
err = [0.012, 0.010, 0.019, 0.009, 0.018]
print(np.mean(Q))
print(np.mean(err))

# 10M
Q = [0.503, 0.642, 0.468, 0.418, 0.325]
err = [0.011, 0.015, 0.021, 0.012, 0.015]
print(np.mean(Q))
print(np.mean(err))
