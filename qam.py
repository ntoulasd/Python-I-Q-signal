import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
f = 1  # Frequency (Hz)
t = np.linspace(0, 2 * np.pi, 1000)  # Time array (2 seconds worth of samples)

# Random Amplitude function: Use uniform distribution for amplitude between 0.5 and 2
A = np.random.uniform(0.5, 2, size=len(t))  # Amplitude changes randomly during time

# I and Q components with random amplitude
I = A * np.cos(2 * np.pi * f * t)  # In-phase component (cosine)
Q = A * np.sin(2 * np.pi * f * t)  # Quadrature component (sine)

# Create 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot I and Q as a 3D spiral with random amplitude
ax.plot(I, Q, t, label='I/Q Signal with Random Amplitude', color='b')

# Labels and title
ax.set_xlabel('In-phase (I)')
ax.set_ylabel('Quadrature (Q)')
ax.set_zlabel('Time (t)')
ax.set_title('3D Spiral of I/Q Signal with Random Amplitude')

# Show the plot
plt.legend()
plt.show()
