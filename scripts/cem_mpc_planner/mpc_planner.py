import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

safety_distance = 5.0

class KinematicBicycleModel:
    def __init__(self, dt, L):
        self.dt = dt
        self.L = L

    def simulate(self, x0, u):
        x = x0.copy()
        for i in range(u.shape[1]):
            x[0] += x[3] * np.cos(x[2]) * self.dt
            x[1] += x[3] * np.sin(x[2]) * self.dt
            x[2] += (x[3] / self.L) * np.tan(u[0, i]) * self.dt
            x[3] += u[1, i] * self.dt
        return x

def cem_mpc(x0, desired_trajectory, car_model, other_vehicle_predictions, horizon, num_samples, num_iterations):
    control_dim = 2
    mean = np.zeros((control_dim, horizon))
    covariance = np.eye(control_dim * horizon)

    for _ in range(num_iterations):
        samples = np.random.multivariate_normal(mean.ravel(), covariance, num_samples).T.reshape(control_dim, horizon, num_samples)
        costs = np.zeros(num_samples)

        for i in range(num_samples):
            x = x0
            for t in range(horizon):
                x = car_model.simulate(x, samples[:, t, i].reshape(-1, 1))
                costs[i] += np.linalg.norm(x[:2] - desired_trajectory[:2, t])  # Position error cost
                costs[i] += 0.01 * np.abs(x[3] - desired_trajectory[3, t])  # Speed error cost
                
                for other_vehicle_pred in other_vehicle_predictions[t]:
                    distance_to_other_vehicle = np.linalg.norm(x[:2] - other_vehicle_pred[:2])
                    if distance_to_other_vehicle < safety_distance:
                        costs[i] += 1000 * (1 / distance_to_other_vehicle)  # Penalize close proximity to other vehicles

        elite_indices = np.argsort(costs)[:num_samples // 5]
        elite_samples = samples[:, :, elite_indices]

        mean = np.mean(elite_samples, axis=-1)
        covariance = np.cov(elite_samples.reshape(control_dim, -1), rowvar=True) + 1e-6 * np.eye(control_dim * horizon)

    return mean[:, 0]

def main():
    dt = 0.1
    L = 2.0
    car_model = KinematicBicycleModel(dt, L)
    x0 = np.array([0, 0, 0, 0])

    desired_trajectory = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ])

    horizon = 5
    num_samples = 500
    num_iterations = 10

    control_input = cem_mpc(x0, desired_trajectory, car_model, horizon, num_samples, num_iterations)
    print("Optimal control input: throttle =", control_input[1], ", steering_angle =", control_input[0])

if __name__ == "__main__":
    main()
