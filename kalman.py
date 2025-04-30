import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
g = 9.8
seconds = 10
n = int(seconds / dt)
x = np.zeros((2, n))
x[:, 0] = np.array([100.0, 0.0]) # we start with position 100m and velocity 0
t = np.linspace(0, seconds, n)

p_variance = [1, 0.5, 0.0] # pos, vel, covariance 
m_variance = [1, 0.5, 0.0]

z = np.zeros((2, n)) # for the measurements
x_estimated = np.zeros((2, n))
x_estimated[:, 0] = np.array([100.0, 0.0]) # we start with position 100m and velocity 0 # let's just assume we know the thing correctly

# runs the simulation of something falling basically
for i in range(1, n):
    x[0, i] = x[0, i - 1] + (x[1, i - 1] * dt) + ((-0.5) * g * dt ** 2) + np.random.normal(0, p_variance[0])
    x[1, i] = x[1, i - 1] + ((-1.0) * g * dt) + np.random.normal(0, p_variance[1])

z[0, :] = x[0, :] + np.random.normal(0, m_variance[0], n)
z[1, :] = x[1, :] + np.random.normal(0, m_variance[1], n)

# assume all covariances/variances constant over time


####### 2D CASE #################
# filter stuff
P = np.eye(2) # this is the moving error matrix essentially
Q = np.matrix([[p_variance[0] ** 2, p_variance[2]],
               [p_variance[2], p_variance[1] ** 2]])
R = np.matrix([[m_variance[0] ** 2, m_variance[2]],
               [m_variance[2], m_variance[1] ** 2]]) # sets up the covariance matriceis
A = np.matrix([[1, dt], [0, 1]])   # this is the state transition matrix
B = np.matrix([[0.5 * dt**2], [dt]]) # this is the control input update where gravity is the input
u = -g # gravity is negative bc falling
H = np.eye(2) # this is the measurement prediction; ie, given state what is the predicted next sensor reading
# implemented filter with both position and velocity as a 2D state
for i in range(1, n):
    x_hat = (A @ x_estimated[:, i - 1].reshape((2, 1))) + (B * u) # state transition
    p_hat = (A @ P @ A.T) + Q # moving error prediction
    
    K = p_hat @ H.T @ np.linalg.inv(H @ p_hat @ H.T + R) # the kalman gain calculation
    r = z[:, i].reshape((2, 1))  - (H @ x_hat) # this is the innovation term

    x_estimated[:, i] = (x_hat + (K @ r)).reshape((2,)) # this (finally) updates the state
    P = (np.eye(2) - (K @ H)) @ p_hat # this updates the moving error
two_d_x_estimated = np.copy(x_estimated)
x_estimated = np.zeros((2, n))
x_estimated[:, 0] = np.array([100.0, 0.0])

##### 1D POSITION CASE ##################
# we assume that all covariances are constant for process/measurement
P = 1.0 # this is the moving error matrix essentially
Q = p_variance[0]
R = m_variance[0] # sets up the covariance matriceis
A = 1.0   # this is the state transition matrix
B = 0.5 * dt**2 # this is the control input update where gravity is the input
u = -g # gravity is negative bc falling
H = 1.0 # this is the measurement prediction; ie, given state what is the predicted next sensor reading
# implemented filter with only position state tracking
for i in range(1, n):
    x_hat = (A * x_estimated[0, i - 1]) + (B * u) # state transition
    p_hat = (A * P * A) + Q # moving error prediction
    
    K = p_hat * H * np.linalg.inv(np.matrix(H * p_hat * H + R)) # the kalman gain calculation
    r = z[0, i]  - (H * x_hat) # this is the innovation term
    
    x_estimated[0, i] = (x_hat + (K[0,0] * r)) # this (finally) updates the state
    P = (1 - (K * H)) * p_hat # this updates the moving error
position_x_estimated = np.copy(x_estimated[0])
x_estimated = np.zeros((2, n))
x_estimated[:, 0] = np.array([100.0, 0.0])

### 1D CASE FOR VELOCITY
# we assume that all covariances are constant for process/measurement
P = 1.0 # this is the moving error matrix essentially
Q = p_variance[1]
R = m_variance[1] # sets up the covariance matriceis
A = 1.0   # this is the state transition matrix
B = 0.5 * dt**2 # this is the control input update where gravity is the input
u = -g # gravity is negative bc falling
H = 1.0 # this is the measurement prediction; ie, given state what is the predicted next sensor reading
# implemented filter with only velocity state tracking
for i in range(1, n):
    x_hat = (A * x_estimated[1, i - 1]) + (B * u) # state transition
    p_hat = (A * P * A) + Q # moving error prediction
    
    K = p_hat * H * np.linalg.inv(np.matrix(H * p_hat * H + R)) # the kalman gain calculation
    r = z[1, i]  - (H * x_hat) # this is the innovation term
    
    x_estimated[1, i] = (x_hat + (K[0,0] * r)) # this (finally) updates the state
    P = (1 - (K * H)) * p_hat # this updates the moving error
velocity_x_estimated = np.copy(x_estimated[1])
x_estimated = np.zeros((2, n))
x_estimated[:, 0] = np.array([100.0, 0.0])

## CHAT GPT WROTE THE MATPLOTLIB CODE!!!!!

# Plot comparison of position estimates
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, x[0, :], label='True Position', linewidth=2)
plt.plot(t, z[0, :], 'o', label='Measured Position', markersize=3, alpha=0.5)
plt.plot(t, two_d_x_estimated[0, :], '--', label='2D Kalman Position')
plt.plot(t, position_x_estimated, '--', label='1D Position Kalman')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Position Estimate Comparison')
plt.legend()
plt.grid(True)

# Plot comparison of velocity estimates
plt.subplot(1, 2, 2)
plt.plot(t, x[1, :], label='True Velocity', linewidth=2)
plt.plot(t, z[1, :], 'o', label='Measured Velocity', markersize=3, alpha=0.5)
plt.plot(t, two_d_x_estimated[1, :], '--', label='2D Kalman Velocity')
plt.plot(t, velocity_x_estimated, '--', label='1D Velocity Kalman')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity Estimate Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print()
# Compare errors
position_errors = {
    '2D': np.mean(np.abs(two_d_x_estimated[0, :] - x[0, :])),
    '1D Position': np.mean(np.abs(position_x_estimated - x[0, :]))
}

velocity_errors = {
    '2D': np.mean(np.abs(two_d_x_estimated[1, :] - x[1, :])),
    '1D Velocity': np.mean(np.abs(velocity_x_estimated - x[1, :]))
}

print("\n=== Average Absolute Errors ===")
print("Position:")
for k, v in position_errors.items():
    print(f"  {k}: {v:.3f} m")

print("Velocity:")
for k, v in velocity_errors.items():
    print(f"  {k}: {v:.3f} m/s")