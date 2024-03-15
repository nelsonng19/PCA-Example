import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file into a DataFrame
df = pd.read_csv('./iris.csv')

# Convert DataFrame to NumPy array
data = df.to_numpy()

x = np.array(data[:, :-1], dtype=float)
x_t = np.transpose(x)
xx = x_t @ x

eigenvalues, eigenvectors = np.linalg.eig(xx)
# print(eigenvalues)
d1 = eigenvectors[:, 2]
d1 = d1[np.newaxis, :]
d2 = eigenvectors[:, 3]
d2 = d2[np.newaxis, :]
d = np.concatenate([d1, d2], axis=0)
PCA = d @ x_t

fig = plt.figure()
ax = fig.add_subplot(111)
# Plot the setosa points
for i in range(50):
    point = PCA[:, i]
    ax.scatter(point[0], point[1], c="red", label="Setosa" if i == 0 else None)

# Plot the versicolor points
for i in range(50, 100):
    point = PCA[:, i]
    ax.scatter(point[0], point[1], c="blue", label="Versicolor" if i == 50 else None)

# Plot the virginica points
for i in range(100, 150):
    point = PCA[:, i]
    ax.scatter(point[0], point[1], c="black", label="Virginica" if i == 100 else None)

ax.legend()
plt.show()
