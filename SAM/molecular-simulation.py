# %% [markdown]
# # **⚡ SAM Algorithm**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## **🧮 Intermolecular Potential**

# %%
# Lennard-Jones potential between all pairs
def intermolecular_pair_potential(coordinates, epsilon=1.0, sigma=1.0):
    total_energy = 0.0
    n = len(coordinates)
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = np.linalg.norm(coordinates[i] - coordinates[j])
            # Avoid division by zero or too small r
            if r == 0:
                continue
            total_energy += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return total_energy

# %% [markdown]
# ## **🧬 Annealing Lennard-Jones**

# %%
# Simulated annealing MCMC optimizer for particle coordinates
def simulated_annealing_lj(initial_coords, epsilon=1.0, sigma=1.0, step_size=0.05, T0=10, alpha=0.999):
    n_particles = initial_coords.shape[0]
    max_iterations = 2000 * n_particles  # Scale samples with particle count

    coords = initial_coords.copy()
    current_energy = intermolecular_pair_potential(coords, epsilon, sigma)
    best_coords = coords.copy()
    best_energy = current_energy

    T = T0
    for iteration in range(max_iterations):
        # Propose new coordinates with small Gaussian perturbation
        proposal = coords + np.random.normal(scale=step_size, size=coords.shape)
        
        # Check if any coordinate goes out of bounds [-3, 3]
        if np.any(proposal > 3) or np.any(proposal < -3):
            # Reinitialize proposal randomly within [-3, 3]
            proposal = np.random.uniform(low=-3, high=3, size=coords.shape)

        proposal_energy = intermolecular_pair_potential(proposal, epsilon, sigma)

        delta_energy = proposal_energy - current_energy

        # Acceptance probability
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
            coords = proposal
            current_energy = proposal_energy

            # Track best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_coords = coords.copy()

        # Temperature decay
        T *= alpha

        # Optional: print progress every 2000 iterations
        if iteration % 2000 == 0:
            print(f"Iter {iteration:5d} | Temp {T:.5f} | Best Energy so far {best_energy:.5f}")

        # Stop early if temperature is very low
        if T < 1e-6:
            break

    print(f"Best energy found: {best_energy:.5f}")
    return best_coords, best_energy

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting even if unused directly

def plot_particles_3d(coords, title="Particle Coordinates in 3D"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', s=50)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    plt.show()


# %% [markdown]
# ## **🧪 Testing**

# %%
# Example usage
np.random.seed(42)
initial_coords = np.random.rand(30, 3)  # 10 particles in 3D

optimized_coords, final_energy = simulated_annealing_lj(initial_coords)

print("Optimized Coordinates:\n", optimized_coords)
print("Final Potential Energy:", final_energy)

plot_particles_3d(optimized_coords, title=f"Optimized Particle Positions (Energy: {final_energy:.3f})")

