# %% [markdown]
# # **⚡ Lennard Jones Potential**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## **🧪 Metric**

# %%
# Metric definition.
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# %% [markdown]
# ## **🧬 Classic Potential**

# %%
# Classic potential.
def classic_potential(r, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


# %% [markdown]
# ## **🧮 Pairwise Potential**

# %%
# Custom intermolecular pairwise potential energy for all.
def intermolecular_pair_potential(coordinates, epsilon=1.0, sigma=1.0):
    sum = 0.0
    n = len(coordinates)

    # First summation.
    for i in range(n - 1):

        # Second summation.
        for j in range(i + 1, n):
            r = euclidean_distance(coordinates[i], coordinates[j])
            sum += classic_potential(r, epsilon, sigma)
    
    return sum

# %% [markdown]
# ## **🚀 Finite Difference**

# %%
def optimize_coordinates(initial_coords, epsilon=1.0, sigma=1.0, lr=0.01, max_iter=100, tol=1e-6):
    coords = initial_coords.copy()

    def energy_wrapper(flat_coords):
        reshaped = flat_coords.reshape((-1, coords.shape[1]))
        return intermolecular_pair_potential(reshaped, epsilon, sigma)

    for it in range(max_iter):
        flat_coords = coords.flatten()
        grad_approx = np.zeros_like(flat_coords)

        # Numerical gradient
        h = 1e-5
        for i in range(len(flat_coords)):
            x1 = flat_coords.copy()
            x2 = flat_coords.copy()
            x1[i] += h
            x2[i] -= h
            grad_approx[i] = (energy_wrapper(x1) - energy_wrapper(x2)) / (2 * h)

        # Update coordinates
        flat_coords -= lr * grad_approx
        coords = flat_coords.reshape(coords.shape)

        # Check convergence
        if np.linalg.norm(grad_approx) < tol:
            print(f"Converged at iteration {it}")
            break

    return coords

# %% [markdown]
# ## **🎲 Normally Distributed Particles**

# %%
np.random.seed(42)
initial_coords = np.random.rand(20, 3)  # 5 particles in 3D
optimized_coords = optimize_coordinates(initial_coords)

print("Optimized Coordinates:")
print(optimized_coords)
print("Potential Energy:", intermolecular_pair_potential(optimized_coords))

# %% [markdown]
# ## **⭐ System**

# %%
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np

coordinates = optimized_coords

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x = coordinates[:, 0]
y = coordinates[:, 1]
z = coordinates[:, 2]

ax.scatter(x, y, z, color='blue', s=100, label='Particles')

# Set axis limits to zoom in
lower_lim = 1e8
upper_lim = -1e8
ax.set_xlim(lower_lim, upper_lim)
ax.set_ylim(lower_lim, upper_lim)
ax.set_zlim(lower_lim, upper_lim)

ax.set_title("Lennard-Jones System")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()



