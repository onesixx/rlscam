import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)  # Set random seed (optional)
iter = 1000

# Generate points within the unit circle using a transformation.
r = np.sqrt(np.random.rand(iter))  # Random radii (0 to 1)
theta = 2 * np.pi * np.random.rand(iter)  # Random angles (0 to 2*pi)
x = r * np.cos(theta)  # x-coordinates
y = r * np.sin(theta)  # y-coordinates

# Calculate the ratio of points inside the circle.
circle_in_count = np.sum(r < 1)

# Calculate the size of the circle using the Monte Carlo estimation.
circle_area = circle_in_count / iter * 4  # Since we're working with a quarter circle, we multiply by 4

# Output result
print(f'=== Iteration : {iter}, Estimated Circle Area = {circle_area}, Estimated Circle Radius = {np.sqrt(circle_area / np.pi)}, Actual Circle Radius = 1.0 ===')

# Visualize points inside and outside the circle.
plt.scatter(x, y, c='b', marker='.')
plt.title('Monte-Carlo Circle Estimation')
plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio for a circle
plt.show()
