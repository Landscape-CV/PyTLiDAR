import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import rpca.ialm

# 1. Create a synthetic cylinder
theta = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(0, 10, 200)
theta, z = np.meshgrid(theta, z)
x = np.cos(theta).ravel()
y = np.sin(theta).ravel()
z = z.ravel()

cylinder = np.vstack((x, y, z)).T  # shape (n, 3)

# 2. Add sparse noise
np.random.seed(42)
sparse_noise = np.zeros_like(cylinder)
num_outliers = 300
indices = np.random.choice(len(cylinder), size=num_outliers, replace=False)
sparse_noise[indices] = 0.5 * np.random.randn(num_outliers, 3)
D = cylinder + sparse_noise

# 3. Apply RPCA (Inexact ALM)
A, E = rpca.ialm.fit(D)

# 4. PCA on the low-rank component
pca = PCA(n_components=3)
pca.fit(A)
direction = pca.components_[0]  # First principal component (unit vector)

# 5. Visualization
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot original noisy point cloud
ax.scatter(D[:, 0], D[:, 1], D[:, 2], color='lightgray', s=1, label='Noisy input')

# Plot RPCA-cleaned low-rank structure
ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='green', s=1, label='Low-rank (cleaned)')

# Plot the principal axis
mean = A.mean(axis=0)
scale = 5
ax.quiver(mean[0], mean[1], mean[2],
          direction[0], direction[1], direction[2],
          color='red', length=scale, linewidth=2, label='Principal Axis')

ax.set_title("RPCA + PCA on a Cylindrical Point Cloud")
ax.legend()


# Extend the principal axis in both directions
t = np.linspace(-5, 5, 100)  # adjust range for visual clarity
line_pts = mean + np.outer(t, direction)

ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
        color='red', linewidth=2, label='Principal Axis')

plt.show()