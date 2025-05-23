import numpy as np

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

# Simulated annealing MCMC optimizer for particle coordinates
def simulated_annealing_lj(initial_coords, epsilon=1.0, sigma=1.0, max_iterations=20000, step_size=0.05, T0=10, alpha=0.999):
    coords = initial_coords.copy()
    current_energy = intermolecular_pair_potential(coords, epsilon, sigma)
    best_coords = coords.copy()
    best_energy = current_energy
    
    T = T0
    for iteration in range(max_iterations):
        # Propose new coordinates with small Gaussian perturbation
        proposal = coords + np.random.normal(scale=step_size, size=coords.shape)
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
            print(f"Iter {iteration:5d} | Temp {T:.5f} | Energy {current_energy:.5f}")
            
        # Stop early if temperature is very low
        if T < 1e-6:
            break
    
    print(f"Best energy found: {best_energy:.5f}")
    return best_coords, best_energy

# Example usage
np.random.seed(42)
initial_coords = np.random.rand(30, 3)  # 10 particles in 3D

optimized_coords, final_energy = simulated_annealing_lj(initial_coords)

print("Optimized Coordinates:\n", optimized_coords)
print("Final Potential Energy:", final_energy)
