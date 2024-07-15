import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data from CSV file
data = pd.read_csv("PF_ultrafinal1.csv", skiprows=1)

# Extracting necessary columns from the dataframe
f_values = data.iloc[:, 0].values
stress_values = data.iloc[:, 4].values
actual_y_values = data.iloc[:, -1].values

# Number of particles
N_particles = 10000

def calculate_weights(predicted_y, actual_y):
    """
    Calculate weights for each particle based on the predicted and actual values.
    """
    weights = actual_y / predicted_y
    return weights / np.sum(weights)  # Normalize weights

def model_output(f, stress, b, s):
    """
    Calculate the model output based on the Steinberg formula.
    """
    # if b < 0:
    #     return ((s**(1/b)) / (2 * f * (stress**(1/b)) * (0.683 + (0.271 * 2**(1/b)) + (0.043 * 3**(1/b)))))
    # else:
    return ((s**(-1/b)) / (2 * f * (stress**(-1/b)) * (0.683 + (0.271 * 2**(-1/b)) + (0.043 * 3**(-1/b))))

def systematic_resampling(weights):
    """
    Perform systematic resampling to select particles based on their weights.
    """
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def estimate_parameters(particles):
    """
    Estimate the parameters by taking the mean of the particles.
    """
    return np.mean(particles, axis=0)

def objective_function(parameters, f_values, stress_values, actual_y_values):
    """
    Calculate the sum of squared errors between predicted and actual values.
    """
    b, s = parameters
    predicted_y = np.array([model_output(f, stress, b, s) for f, stress in zip(f_values, stress_values)])
    error = np.sum((predicted_y - actual_y_values) ** 2)
    return error

def hybrid_particle_filter_nelder_mead(data, N_particles=1000):
    """
    Hybrid particle filter with Nelder-Mead optimization for parameter estimation.
    """
    # Initialization (you can modify this based on your preferred particle filter method)
    b_range = (-2, -0.1)
    s_range = (100, 200)
    particles = np.zeros((N_particles, 2))
    particles[:, 0] = np.random.uniform(b_range[0], b_range[1], N_particles)  # b values
    particles[:, 1] = np.random.uniform(s_range[0], s_range[1], N_particles)  # s values
    weights = np.ones(N_particles) / N_particles
    
    for f, stress, actual_y in zip(data.iloc[:, 0], data.iloc[:, 4], data.iloc[:, -1]):
        # Prediction
        predicted_y = np.array([model_output(f, stress, particle[0], particle[1]) for particle in particles])
        
        # Correction
        weights = calculate_weights(predicted_y, actual_y)
        
        # Resampling
        indexes = systematic_resampling(weights)
        particles = particles[indexes]
        
        # Estimation (optional per iteration)
        b_est, s_est = estimate_parameters(particles)
        
    # Final parameter estimation using particle filter
    b_pf, s_pf = estimate_parameters(particles)

    # Refine estimate using Nelder-Mead optimization
    result = minimize(objective_function, 
                      x0=[b_pf, s_pf],  # Start optimization from particle filter estimate 
                      args=(data.iloc[:, 0], data.iloc[:, 4], data.iloc[:, -1]), 
                      method='Nelder-Mead')

    if result.success:
        return result.x  # Return the optimized parameters
    else:
        print("Optimization failed to converge.")
        return b_pf, s_pf  # Fall back to the particle filter estimate

# Call the hybrid particle filter function with the loaded data
estimated_parameters = hybrid_particle_filter_nelder_mead(data)

# Output the estimated parameters
print("Estimated parameters (b, s) after hybrid optimization:", estimated_parameters)
