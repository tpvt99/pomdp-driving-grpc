#!/usr/bin/env python2

import os, sys, glob
import numpy as np
import math
import time

summit_root = os.path.expanduser("~/summit/")
api_root = os.path.expanduser("~/summit/PythonAPI")
summit_connector_path = os.path.expanduser('~/catkin_ws/src/summit_connector/src/')

try:
    sys.path.append(glob.glob(api_root + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(os.path.basename(__file__) + ": Cannot locate the CARLA egg file!!!")
    sys.exit()

sys.path.append(summit_connector_path)
from pathlib2 import Path

import carla
import summit
import random
from basic_agent import BasicAgent

DATA_PATH = Path(summit_root)/'Data'

def draw_waypoints(waypoints, world, color=carla.Color(255, 0, 0), life_time=50.0):

    for i in range(len(waypoints) - 1):
        world.debug.draw_line(
            carla.Location(waypoints[i].x, waypoints[i].y, 0.0),
            carla.Location(waypoints[i + 1].x, waypoints[i + 1].y, 0.0),
            2,
            color,
            life_time)


class CrossEntropyMPC:
    def __init__(self, iterations, horizon, num_samples, num_elites, desired_trajectories, weights, dt):
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.desired_trajectories = desired_trajectories
        self.dt = dt
        self.iterations = iterations
        self.weights = weights

    # def update_bounding_box_corners(current_state, delta_position, delta_yaw):
    #     '''
    #     Update the bounding box corners of the vehicle based on its current state, change in position, and change in yaw.

    #     Parameters:
    #     current_state (np.array): The current state of the vehicle, which contains the bounding box corner positions.
    #                             Shape: [x, y, yaw, v.x, v.y, top-left corner, top-right corner, bottom-right corner, bottom-left corner]
    #     delta_position (np.array): The change in position (x, y) of the vehicle.
    #                             Shape: [delta_x, delta_y]
    #     delta_yaw (float): The change in yaw of the vehicle.

    #     Returns:
    #     np.array: The updated bounding box corners of the vehicle.
    #             Shape: [updated top-left corner, updated top-right corner, updated bottom-right corner, updated bottom-left corner]
    #     '''

    #     # Extract the current bounding box corners
    #     bb_corners = current_state[5:].reshape(4, 2)

    #     # Create a rotation matrix for the delta_yaw
    #     rot_matrix = np.array([
    #         [np.cos(delta_yaw), -np.sin(delta_yaw)],
    #         [np.sin(delta_yaw), np.cos(delta_yaw)]
    #     ])

    #     # Update the bounding box corners by applying the rotation and translation
    #     updated_bb_corners = np.dot(bb_corners, rot_matrix.T) + delta_position

    #     return updated_bb_corners


    def optimize(self, ego_state, exo_states, exo_predictions):
        '''
        Optimize the control inputs for the ego vehicle using the cross-entropy method.
            ego_state = [x, y, yaw, v.x, v.y, bb_width, bb_length]
            exo_states = dict[exo_id] = [x, y, yaw, v.x, v.y, bb_width, bb_length]
            exo_predictions = dict[exo_id] = [(x, y), (x, y), ...]
        '''
        mean = np.zeros((self.horizon, 2))
        covariance = np.tile(np.eye(2), (self.horizon, 1, 1))

        for _ in range(self.iterations):
            actions = self._sample_actions(mean, covariance)
            costs = np.zeros(self.num_samples)

            for i, action_seq in enumerate(actions):
                current_state = ego_state.copy()
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

                    #next_state[5:] = self.update_bounding_box_corners(current_state, delta_position, delta_yaw)

                    costs[i] += self._cost(next_state, action, self.desired_trajectories[t], exo_states, exo_predictions)
                    current_state = next_state

            elites_indices = np.argsort(costs)[:self.num_elites]
            elites = actions[elites_indices]

            mean, covariance = self._update_distribution(elites)

        optimal_action = mean[0]  # We only apply the first action in the sequence
        return optimal_action

    def _check_collision(self, ego_state, exo_pos_at_t, exo_state_at_0):
        def _rotated_rect_corners(center, width, length, angle):
            half_width = width / 2
            half_length = length / 2
            corners = [
                np.array([-half_width, -half_length]),
                np.array([half_width, -half_length]),
                np.array([half_width, half_length]),
                np.array([-half_width, half_length])
            ]

            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

            return [center + rotation_matrix @ corner for corner in corners]

        def _project(corners, axis):
            projections = [np.dot(corner, axis) for corner in corners]
            return min(projections), max(projections)

        def _overlap(min1, max1, min2, max2):
            return max1 >= min2 and max2 >= min1

        # Compute the yaw of the exogenous vehicle at time t based on its position
        direction_vector = exo_pos_at_t - exo_state_at_0[:2]
        exo_yaw_at_t = np.arctan2(direction_vector[1], direction_vector[0])

        # Compute the corners for both rectangles
        ego_corners = _rotated_rect_corners(ego_state[:2], ego_state[5], ego_state[6], ego_state[2])
        exo_corners = _rotated_rect_corners(exo_pos_at_t, exo_state_at_0[5], exo_state_at_0[6], exo_yaw_at_t)

        # Calculate the edges for both rectangles
        ego_edges = [ego_corners[i] - ego_corners[i - 1] for i in range(len(ego_corners))]
        exo_edges = [exo_corners[i] - exo_corners[i - 1] for i in range(len(exo_corners))]

        # Check for overlap along each axis
        for edge in ego_edges + exo_edges:
            axis = np.array([-edge[1], edge[0]])
            ego_min_proj, ego_max_proj = _project(ego_corners, axis)
            exo_min_proj, exo_max_proj = _project(exo_corners, axis)

            if not _overlap(ego_min_proj, ego_max_proj, exo_min_proj, exo_max_proj):
                return False

        # All axes have overlap, so the rectangles are colliding
        return True


    def _get_bounding_box_corners(self, state):
        """
        Given the current state of the ego vehicle, return its bounding box corners.

        Parameters:
            state (np.array): Current state of the ego vehicle.
                Shape: [x, y, yaw, vel.x, vel.y, width, length]

        Returns:
            np.array: The corners of the bounding box.
                Shape: [top-left corner, top-right corner, bottom-right corner, bottom-left corner]
        """

        loc = np.array([state[0], state[1]])
        yaw = state[2]
        v_x = state[3]
        v_y = state[4]
        width = state[5]
        length = state[6]

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        sideward_vec = np.array([-forward_vec[1], forward_vec[0]])

        half_x_len_forward = 0.5 * length
        half_x_len_backward = -0.5 * length
        half_y_len = 0.5 * width

        corners = [loc - half_x_len_backward * forward_vec + half_y_len * sideward_vec,
                loc + half_x_len_forward * forward_vec + half_y_len * sideward_vec,
                loc + half_x_len_forward * forward_vec - half_y_len * sideward_vec,
                loc - half_x_len_backward * forward_vec - half_y_len * sideward_vec]

        return corners
    

    # def _check_collision(self, ego_state_at_t, exo_pred_at_t, exo_state_at_0):
    #     ego_bb = self._get_bounding_box_corners(ego_state_at_t)
        
    #     # Extract width and length of exo vehicle from exo state at time 0
    #     exo_width = exo_state_at_0[5]
    #     exo_length = exo_state_at_0[6]
        
    #     # Calculate the bounding box corners of the exo vehicle at time t using its prediction and dimensions
    #     exo_loc_at_t = exo_pred_at_t
    #     exo_yaw_at_t = exo_state_at_0[2]  # Assuming constant yaw
    #     exo_forward_vec = np.array([np.cos(exo_yaw_at_t), np.sin(exo_yaw_at_t)])
    #     exo_sideward_vec = np.array([np.sin(exo_yaw_at_t), -np.cos(exo_yaw_at_t)])
    #     exo_half_x_len_forward = exo_length / 2.0
    #     exo_half_x_len_backward = exo_length / 2.0
    #     exo_half_y_len = exo_width / 2.0
    #     exo_corners = [
    #         exo_loc_at_t - exo_half_x_len_backward * exo_forward_vec + exo_half_y_len * exo_sideward_vec,
    #         exo_loc_at_t + exo_half_x_len_forward * exo_forward_vec + exo_half_y_len * exo_sideward_vec,
    #         exo_loc_at_t + exo_half_x_len_forward * exo_forward_vec - exo_half_y_len * exo_sideward_vec,
    #         exo_loc_at_t - exo_half_x_len_backward * exo_forward_vec - exo_half_y_len * exo_sideward_vec
    #     ]
        
    #     # Check collision between the two bounding boxes
    #     for corner in ego_bb:
    #         if np.min(exo_corners[:, 0]) <= corner[0] <= np.max(exo_corners[:, 0]) and np.min(exo_corners[:, 1]) <= corner[1] <= np.max(exo_corners[:, 1]):
    #             return True
        
    #     return False

    
    def _cost(self, next_state, action, desired_trajectory, exo_states, exo_predictions, current_t):
        '''
            Compute the cost of the next state and action.
            next_state = np.array([x, y, yaw, v.x, v.y, bb_width, bb_length])
            action = np.array([acceleration, yaw_rate])
            desired_trajectory = np.array([[x, y], [x, y], ...])
            exo_states = dict[exo_id] = [x, y, yaw, v.x, v.y, bb_width, bb_length]
            exo_predictions = dict[exo_id] = [(x, y), (x, y), ...]
            current_t : current time step
            We can access desirec_trajectory[current_t] to get the desired trajectory at current time step
            Also we can access exo_states[exo_id][t] to get the state of exo vehicle at current time step
        '''
        # Calculate the tracking error cost
        tracking_error = np.linalg.norm(next_state[:2] - desired_trajectory[current_t])
        tracking_cost = self.weights[0] * tracking_error

        # Calculate the control effort cost
        control_effort_cost = self.weights[1] * (action[0]**2 + action[1]**2)

        # Calculate the collision cost
        collision_cost = 0
        for exo_id, exo_pred in exo_predictions.items():
            exo_state_at_0 = exo_states[0] # To know the initial position, heading, and bbox to find the bbox of future timestep
            exo_pred_at_t = exo_pred[current_t] # To know the future position
            if self._check_collision(next_state, exo_pred_at_t, exo_state_at_0):
                collision_cost = self.weights[2]
                break

        # Calculate the total cost
        total_cost = tracking_cost + control_effort_cost + collision_cost

        return total_cost

        
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



client = carla.Client('localhost', 15452)
client.set_timeout(2.0)
world = client.get_world()

map_location = 'meskel_square'

with (DATA_PATH/'{}.sim_bounds'.format(map_location)).open('r') as f:
    bounds_min = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
    bounds_max = carla.Vector2D(*[float(v) for v in f.readline().split(',')])

sumo_network = carla.SumoNetwork.load(str(DATA_PATH/'{}.net.xml'.format(map_location)))
sumo_network_segments = sumo_network.create_segment_map()
sumo_network_spawn_segments = sumo_network_segments.intersection(carla.OccupancyMap(bounds_min, bounds_max))
sumo_network_spawn_segments.seed_rand(42)
sumo_network_occupancy = carla.OccupancyMap.load(str(DATA_PATH/'{}.network.wkt'.format(map_location)))

def rand_path(sumo_network, min_points, interval, segment_map, min_safe_points=None, rng=random):
    if min_safe_points is None:
        min_safe_points = min_points

    spawn_point = None
    route_paths = None
    while not spawn_point or len(route_paths) < 1:
        spawn_point = segment_map.rand_point()
        spawn_point = sumo_network.get_nearest_route_point(spawn_point)
        route_paths = sumo_network.get_next_route_paths(spawn_point, min_safe_points - 1, interval)

    return rng.choice(route_paths)[0:min_points]

def get_position(path_point):
    return sumo_network.get_route_point_position(path_point)

def get_yaw(path_point, path_point_next):
    pos = sumo_network.get_route_point_position(path_point)
    next_pos = sumo_network.get_route_point_position(path_point_next)
    return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x))

path = rand_path(
    sumo_network, 200, 1.0, sumo_network_spawn_segments,
    min_safe_points=100, rng=random.Random(42))

print("Path[0]: ", path[0], " position: ", get_position(path[0]))
# Creating carla waypoints

values = [get_position(path[i]) for i in range(len(path) - 1)]
print(values[0]) # A vector2D
draw_waypoints(values, world, color=carla.Color(r=0, g=255, b=0), life_time=20)

# We first reset the world
actor_list = world.get_actors()

# Find the actor with name "model3"
for actor in actor_list:
    if actor.type_id == "vehicle.tesla.model3":
        actor.destroy()
        break

# query for the cars blueprint.
vehicle_blueprint = client.get_world().get_blueprint_library().filter('model3')[0]

# We now need to obtain a spawn location.
spawn_point = values[10] # spawn at 10th position
spawn_trans = carla.Transform()
spawn_trans.location.x = spawn_point.x
spawn_trans.location.y = spawn_point.y
spawn_trans.location.z = 0.5
spawn_trans.rotation.yaw = get_yaw(path[10], path[11])
vehicle = client.get_world().try_spawn_actor(vehicle_blueprint, spawn_trans)

print('inspection')
print(vehicle.get_location())
print(vehicle.get_transform())
print(spawn_trans)
### Now is the control part


# Set up the CrossEntropyMPC
horizon = 10
num_samples = 1000
num_elites = 50
desired_trajectories = np.array([[val.x, val.y] for val in values])
dt = 0.3
iterations = 100
weights = np.array([1.0, 1.0, 1.0])
mpc = CrossEntropyMPC(iterations, horizon, num_samples, num_elites, desired_trajectories, weights, dt)

# Main control loop
try:
    while True:
        # Get the current state of the ego vehicle
        ego_transform = vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        ego_velocity = vehicle.get_velocity()
        width, length = 1,2

        vehicle_state = np.array([
            ego_location.x,
            ego_location.y,
            np.radians(ego_rotation.yaw),
            ego_velocity.x,
            ego_velocity.y,
            width,
            length
        ])

        # Generate predictions for other vehicles
        # other_vehicles = world.get_actors().filter("vehicle.*")
        # other_vehicles_prediction = []

        # for other_vehicle in other_vehicles:
        #     if other_vehicle.id == ego_vehicle.id:
        #         continue

        #     other_transform = other_vehicle.get_transform()
        #     other_location = other_transform.location
        #     other_velocity = other_vehicle.get_velocity()

        #     other_vehicles_prediction.append([
        #         other_location.x,
        #         other_location.y,
        #         other_velocity.x,
        #         other_velocity.y
        #     ])

        #other_vehicles_prediction = np.array(other_vehicles_prediction)

        # Optimize control inputs using CrossEntropyMPC
        optimal_action = mpc.optimize(vehicle_state, [], [])

        # Apply control inputs to the ego vehicle
        control = carla.VehicleControl()
        control.throttle = max(0, min(1, optimal_action[0]))
        control.steer = max(-1, min(1, optimal_action[1]))
        vehicle.apply_control(control)

        # Wait for dt seconds
        time.sleep(dt)

finally:
    vehicle.destroy()


