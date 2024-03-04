
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from adasghmc import AdaSGHMC

# Simple model with a single parameter
class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.param = nn.Parameter(torch.tensor(0.0))  # Initialize at 0

# Gaussian energy function as loss
def gaussian_energy_loss(x, mu=0, sigma=1):
    return 0.5 * ((x - mu) / sigma) ** 2


def test_psgld_density_matching():
    model = GaussianModel()
    optimizer = AdaSGHMC(model.parameters(), 
                    learning_rate=1e-2, gradient_ema=0.999, 
                    momentum=0.9, eps=1e-8)

    samples = []
    for _ in range(50000):  # Generate more samples for density estimation
        optimizer.zero_grad()
        loss = gaussian_energy_loss(model.param)
        loss.backward()
        optimizer.step()
        
        # Record the current parameter value
        samples.append(model.param.item())
        #print(model.param.item())

    # Convert samples list to a NumPy array for processing
    samples = np.array(samples)[10000:]

    # Bin the samples
    num_bins = 30
    histogram, bins = np.histogram(samples, bins=num_bins, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Calculate expected density
    mu, sigma = 0.0, 1.0  # Gaussian parameters
    expected_density = norm.pdf(bin_centers, mu, sigma)

    # Plotting observed vs expected densities
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, histogram, width=bin_centers[1] - bin_centers[0], color='blue', alpha=0.5, label='Observed Density')
    plt.plot(bin_centers, expected_density, color='red', label='Expected Density')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('PSGLD Sampler Density Estimation')
    plt.show()

# Run the extended test
test_psgld_density_matching()