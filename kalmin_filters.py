import numpy as np

# state vectore:
X = np.array([[0], [0]])

# state transition matrix:
F = np.array([[1, 1], [0, 1]])

# measurment matrix:
H = np.array([[1, 0]])

# ---uncertainties---
# np.eye(2), creates a 2x2 matirx, which is p = [1, 0],[0, 1], then [500, 0],[0, 500]
P = (
    np.eye(2) * 500
)  # uncertainty about position and velocity, 500 = humble uncertainty.
Q = np.eye(2)  # uncertianty about movement, movement noise
R = np.array(
    [[5]]
)  # uncertianty about measurments from sensor, 5 means trust the sensor bu 5 units.

# simulation of noisy measurements (object moving +2 units per step)
measurements = [0, 2.5, 4.1, 6.2, 8.05, 9.9]

# loop over measurements to simulate movement:
for z in measurements:
    # --- predict ---
    X = F @ X
    P = F @ P @ F.T + Q

    # ---update---
    y = np.array([[z]]) - (H @ X)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    X = X + K @ y
    P = (np.eye(2) - K @ H) @ P

    print("Estimate:", X.ravel())
