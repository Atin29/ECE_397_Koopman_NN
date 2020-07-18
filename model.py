import mat4py as mat4py
import scipy.io as sio
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pydmd import DMD
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.metrics import mae, accuracy
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigs

# load matlab data into dictionary
mat_dict = sio.loadmat('sample_data_smib.mat')

# loading x, y matrices
x_t = np.array([mat_dict['X'][i] for i in range(2)])
y_t = np.array([mat_dict['Y'][i] for i in range(2)])

# Koopman Operator K is optimized on |Y-KX|^2 = 0 therefore K = Y/X
k_true = y_t/x_t

# reshape x_in_2d into 2-input numpy array of length 200000
x_in = np.vstack((x_t, y_t))

x_in_rs = np.zeros(400000,)
x_in_rs = x_in_rs.reshape(100000, 4)

for i in range(100000):
    x_in_rs[i] = [x_in[0][i], x_in[1][i], x_in[2][i], x_in[3][i]]
    
y_true = np.zeros(200000,)
y_true = y_true.reshape(100000, 2)

for j in range(100000):
    y_true[j] = [k_true[0][j], k_true[1][j]]

# create training set, must be sequential
# only using first 10k samples to avoid kernel crash
sample_size = 10000
train_ratio = 0.7
train_size = 7000
test_ratio = 0.3
test_size = 3000

x_train, x_test = x_in_rs[:train_size], x_in_rs[(train_size):(sample_size)]
y_train, y_test = y_true[:train_size], y_true[(train_size):(sample_size)]

data_dim = 4
timesteps = 1
data_out = 2

# Tensorflow model
model = Sequential()
# Input Layer
model.add(LSTM(32, activation = 'tanh', return_sequences=True, input_shape=(timesteps, data_dim)))
# model.add(Dense(32, kernel_initializer='normal', activation='relu', input_dim=data_dim))
# Hidden Layer
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
# Hidden Layer
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
# Hidden Layer
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
# The Output Layer :
model.add(Dense(data_out, kernel_initializer='normal',activation='linear'))

# Compile the network :
model.compile(loss='mae', optimizer='adam', metrics=[mae])
model.summary()

# reshape x_train, x_test to fit input in LSTM layers
X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

%%time
model.fit(x = X_train, y = y_train,
                    epochs=5,
                    steps_per_epoch=10)

predictions = model.predict(X_test)
predictions = predictions.reshape(test_size,2)
koopman_pred = predictions.T

dmd = DMD(svd_rank=2)
dmd.fit(koopman_pred)
dmd_eigs = dmd.eigs
dmd_modes = dmd.modes
dmd_dynamics = dmd.dynamics

x = dmd_dynamics[0]
y = dmd_dynamics[1]

fig = plt.figure()
ax = fig.gca(projection='3d')

# PLOTTING
print(ax.plot_trisurf(range(3000), x, y))
print(plt.contour(dmd_modes))
print(plt.contour(dmd_dynamics))

x_1 = x_test.T[0]
x_2 = x_test.T[1]
x_0 = np.vstack((x_1, x_2))
eig_func = x_0.T/dmd_modes[0]
ef = eig_func.T
print(ef)
print(plt.contour(ef))
fig = plt.figure()
ax = fig.gca(projection='3d')
print(ax.plot_trisurf(range(3000), ef[0], ef[1]))

dmd = DMD(svd_rank=2)
dmd.fit(y_test.T)
dmd_eigs = dmd.eigs
dmd_modes = dmd.modes
dmd_dynamics = dmd.dynamics
print(dmd_eigs)
print(dmd_eigs.shape)
print(dmd_modes)
print(dmd_modes.shape)
print(dmd_dynamics)
print(dmd_dynamics.shape)

x_1 = x_test.T[0]
x_2 = x_test.T[1]
x_0 = np.vstack((x_1, x_2))
eig_func = x_0.T/dmd_modes[0]
ef = eig_func.T
print(ef)
print(plt.contour(ef))
fig = plt.figure()
ax = fig.gca(projection='3d')
print(ax.plot_trisurf(range(3000), ef[0], ef[1]))

AAT = predictions.dot(koopman_pred)
print(AAT.shape)
print(AAT)
ATA = koopman_pred.dot(predictions)
print(ATA.shape)
print(ATA)

# single value decomposition of 2D Koopman Vector into Square Matrix
eig_vals, eig_vecs = np.linalg.eig(AAT)
print(eig_vals.shape)
print(eig_vals)
print(eig_vecs.shape)
print(eig_vecs)
eig_vals_T, eig_vecs_T = np.linalg.eig(ATA)
print(eig_vals_T.shape)
print(eig_vals_T)
print(eig_vecs_T.shape)
print(eig_vecs_T)

# single value decomposition of 2D Koopman Vector into Square Matrix
u, s, v = np.linalg.svd(koopman_pred)

X = u
Y = v
plt.plot(X, Y)
plt.show

eigval_sort = np.sort(eigenvalues)
largest_eigval = eigval_sort[test_size-1]
second_largest_eigval = eigval_sort[test_size-2]
third_largest_eigval = eigval_sort[test_size-3]
print("largest_eigenvalue: ", largest_eigval)
print("second_largest_eigenvalue: ", second_largest_eigval)
print("third_largest_eigenvalue: ", third_largest_eigval)

largest_index = 0
second_largest_index = 0
third_largest_index = 0

for i in range(test_size):
    if(largest_eigval == eigenvalues[i]):
        largest_index = i
    if(second_largest_eigval == eigenvalues[i]):
        second_largest_index = i
    if(third_largest_eigval == eigenvalues[i]):
        third_largest_index = i

largest_eigenfunction = eigenvectors[largest_index]
second_largest_eigenfunction = eigenvectors[second_largest_index]
third_largest_eigenfunction = eigenvectors[third_largest_index]

print("largest_eigenfunction\n", largest_eigenfunction)
print("second_largest_eigenfunction\n", second_largest_eigenfunction)
print("third_largest_eigenfunction\n", third_largest_eigenfunction)

origin = [0, 0]
eig_vec1 = largest_eigenfunction
eig_vec2 = second_largest_eigenfunction
eig_vec3 = third_largest_eigenfunction
plt.quiver(*eig_vec1, color=['r'], scale = 0.05)
plt.quiver(*eig_vec2, color=['b'], scale = 0.005)
plt.quiver(*eig_vec3, color=['g'], scale = 0.005)
plt.show()