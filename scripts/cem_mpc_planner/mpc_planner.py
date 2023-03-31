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


import numpy as np
from collections import deque

class CrossEntropyMPC:
    def __init__(self, iterations, horizon, num_samples, num_elites, desired_trajectories, dt):
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.desired_trajectories = desired_trajectories
        self.dt = dt
        self.iterations = iterations

    def optimize(self, ego_state, exo_states, exo_predictions):
        '''
        Optimize the control inputs for the ego vehicle using the cross-entropy method.
            vehicle_state = [x, y, yaw, v.x, v.y, top-left corner, top-right corner, bottom-right corner, bottom-left corner]
            exo_states = dict[exo_id] = [x, y, yaw, v.x, v.y, top-left corner, top-right corner, bottom-right corner, bottom-left corner]
            exo_predictions = dict[exo_id] = [(x, y), (x, y), ...]
        '''
        mean = np.zeros((self.horizon, 2))
        covariance = np.tile(np.eye(2), (self.horizon, 1, 1))

        for _ in range(self.iterations):
            # At each iterations, we got a set of actions sequences and evaluate them
            actions = self._sample_actions(mean, covariance)
            costs = np.zeros(self.num_samples)

            for i, action_seq in enumerate(actions):
                current_state = vehicle_state.copy()
                for t, action in enumerate(action_seq):
                    delta_position = np.array([
                        current_state[3] * self.dt + 0.5 * action[0] * np.cos(current_state[2]) * self.dt ** 2,
                        current_state[4] * self.dt + 0.5 * action[0] * np.sin(current_state[2]) * self.dt ** 2,
                    ])
                    delta_yaw = action[1] * self.dt

                    next_state = current_state.copy()
                    next_state[:5] += np.array([
                        delta_position[0],
                        delta_position[1],
                        delta_yaw,
                        action[0] * np.cos(current_state[2]) * self.dt,
                        action[0] * np.sin(current_state[2]) * self.dt
                    ])
                    next_state[5:] = update_bounding_box_corners(current_state, delta_position, delta_yaw)

                    costs[i] += self._cost(next_state, action, self.desired_trajectories[t], exo_states, exo_predictions)
                    current_state = next_state

            elites_indices = np.argsort(costs)[:self.num_elites]
            elites = actions[elites_indices]

            mean, covariance = self._update_distribution(elites)

        optimal_action = mean[0]  # We only apply the first action in the sequence
        return optimal_action


    
    def _cost(self, next_state, action, desired_trajectory, other_vehicles_prediction):
        # Compute the deviation from the desired trajectory
        deviation_cost = np.sum((next_state[:2] - desired_trajectory) ** 2)

        # Compute the collision cost with other vehicles
        collision_cost = 0
        ego_bounding_box = next_state[5:].reshape(-1, 2)
        for other_vehicle_prediction in other_vehicles_prediction:
            other_bounding_box = other_vehicle_prediction[5:].reshape(-1, 2)
            if _check_collision(ego_bounding_box, other_bounding_box):
                collision_cost += np.inf

        # Add an action cost to penalize high control inputs
        action_cost = np.sum(action ** 2)

        return deviation_cost + collision_cost + action_cost

        
    def _sample_actions(self, mean, covariance):
        """
        Generate random action sequences based on the mean and covariance.

        num_samples: The number of action sequences generated and evaluated at each iteration. 
                More samples can help explore a larger solution space, but it may also increase the computational complexity.
        horizon: The number of time steps considered in the MPC approach. 
                    A longer horizon allows the algorithm to plan further into the future, but it also increases the computational complexity.
        num_acts: The number of control actions for the vehicle. 
            In our case, we have two control actions: acceleration (or throttle) and steering angle.

        Args:
        mean (numpy.ndarray): The mean of the action distribution with shape (horizon, 2).
        covariance (numpy.ndarray): The covariance of the action distribution with shape (horizon, 2, 2).

        Returns:
        numpy.ndarray: A set of random action sequences with shape (num_samples, horizon, 2).
        """
        action_sequences = np.zeros((self.num_samples, self.horizon, 2))

        for i in range(self.horizon):
            action_sequences[:, i, :] = np.random.multivariate_normal(mean[i], covariance[i], self.num_samples)

        return action_sequences

    def _update_distribution(self, elites):
        """
        Update the mean and covariance of the action distribution based on the elite action sequences.

        Args:
        elites (numpy.ndarray): The elite action sequences with shape (num_elites, horizon, 2).

        Returns:
        tuple: A tuple containing the updated mean and covariance of the action distribution.
            The mean has shape (horizon, 2), and the covariance has shape (horizon, 2, 2).
        """
        elite_mean = np.mean(elites, axis=0)
        elite_cov = np.zeros((self.horizon, 2, 2))

        for i in range(self.horizon):
            centered_elites = elites[:, i, :] - elite_mean[i]
            elite_cov[i] = np.dot(centered_elites.T, centered_elites) / self.num_elites

        return elite_mean, elite_cov



def main():
    import carla
    import numpy as np
    import time

    # Connect to the Carla server
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    # Load a map
    world = client.get_world()
    map = world.get_map()

    # Spawn an ego vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = map.get_spawn_points()[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Set up the CrossEntropyMPC
    horizon = 10
    num_samples = 1000
    num_elites = 50
    desired_trajectories = np.array([[x, y] for x, y in zip(range(100), range(100))])
    dt = 0.1
    mpc = CrossEntropyMPC(horizon, num_samples, num_elites, desired_trajectories, dt)

    # Main control loop
    try:
        while True:
            # Get the current state of the ego vehicle
            ego_transform = ego_vehicle.get_transform()
            ego_location = ego_transform.location
            ego_rotation = ego_transform.rotation
            ego_velocity = ego_vehicle.get_velocity()

            vehicle_state = np.array([
                ego_location.x,
                ego_location.y,
                np.radians(ego_rotation.yaw),
                ego_velocity.x,
                ego_velocity.y
            ])

            # Generate predictions for other vehicles
            other_vehicles = world.get_actors().filter("vehicle.*")
            other_vehicles_prediction = []

            for other_vehicle in other_vehicles:
                if other_vehicle.id == ego_vehicle.id:
                    continue

                other_transform = other_vehicle.get_transform()
                other_location = other_transform.location
                other_velocity = other_vehicle.get_velocity()

                other_vehicles_prediction.append([
                    other_location.x,
                    other_location.y,
                    other_velocity.x,
                    other_velocity.y
                ])

            other_vehicles_prediction = np.array(other_vehicles_prediction)

            # Optimize control inputs using CrossEntropyMPC
            optimal_action = mpc.optimize(vehicle_state, other_vehicles_prediction)

            # Apply control inputs to the ego vehicle
            control = carla.VehicleControl()
            control.throttle = max(0, min(1, optimal_action[0]))
            control.steer = max(-1, min(1, optimal_action[1]))
            ego_vehicle.apply_control(control)

            # Wait for dt seconds
            time.sleep(dt)

    finally:
        ego_vehicle.destroy()

if __name__ == "__main__":
    main()
