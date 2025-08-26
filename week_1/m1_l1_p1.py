# Vector - Error Measures

import numpy as np
np.set_printoptions(precision=4) #double precision

y_data = np.array([[20,21,30,30,21,25,38,37,30,22,22,38,20,35]],dtype=np.double).T # [[]] is for 2D array
y_pred = np.array([[21,21,31,30,20,28,36,32,31,20,21,39,21,34]],dtype=np.double).T

print("y_data = \n", y_data)
print("y_pred = \n", y_pred)

y_err = y_data - y_pred

print("y_err = \n", y_err)
print("Size of y_err:", np.shape(y_err))

# Mean Squared Error (MSE)
MSE_loop = 0
for i in range(len(y_err)):
    MSE_loop += (y_err[i]) ** 2
MSE_loop /= len(y_data)
print("MSE (loop) =", MSE_loop)

# Mean Squared Error (MSE) using matrix multiplication
MSE_mm = (y_err.T @ y_err) / len(y_data)
print("MSE (loop) = ", MSE_loop)

# Mean Squared Error (MSE) using np
MSE_np = np.mean(np.square(y_err))
print(f"MSE (Numpy) = {MSE_np:.4f}")