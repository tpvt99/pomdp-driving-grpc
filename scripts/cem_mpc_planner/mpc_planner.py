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

DATA_PATH = Path(summit_root)/'Data'

def draw_waypoints(waypoints, world, color=carla.Color(255, 0, 0), life_time=50.0):

    for i in range(len(waypoints) - 1):
        world.debug.draw_line(
            carla.Location(waypoints[i].x, waypoints[i].y, 0.0),
            carla.Location(waypoints[i + 1].x, waypoints[i + 1].y, 0.0),
            2,
            color,
            life_time)

import numpy as np

def bicycle_model(current_state, action, dt):
    x, y, yaw, vel_x, vel_y = current_state
    acceleration, steering_angle = action

    # Calculate the scalar speed
    speed = np.sqrt(vel_x**2 + vel_y**2)

    # Update the position and yaw
    x += speed * np.cos(yaw) * dt
    y += speed * np.sin(yaw) * dt
    yaw += speed * np.tan(steering_angle) / L * dt  # L is the wheelbase of the vehicle

    # Ensure yaw is within -pi to pi range
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    # Update the longitudinal and lateral velocities
    vel_x = speed * np.cos(yaw) + acceleration * np.cos(yaw) * dt
    vel_y = speed * np.sin(yaw) + acceleration * np.sin(yaw) * dt

    next_state = np.array([x, y, yaw, vel_x, vel_y])
    return next_state


class CrossEntropyMPC:
    def __init__(self, iterations, horizon, num_samples, num_elites, desired_trajectories, weights, dt, L):
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.desired_trajectories = desired_trajectories
        self.dt = dt
        self.iterations = iterations
        self.weights = weights
        self.desired_speed = 6.0
        self.L = L


    def optimize(self, ego_state, exo_states, exo_predictions):
        '''
        Optimize the control inputs for the ego vehicle using the cross-entropy method.
            ego_state = [x, y, yaw, v.x, v.y, bb_width, bb_length]
            exo_states = dict[exo_id] = [x, y, yaw, v.x, v.y, bb_width, bb_length]
            exo_predictions = dict[exo_id] = [(x, y), (x, y), ...]
        '''
        mean = np.zeros((self.horizon, 2))
        #mean[:, 0] = self.desired_speed/2
        covariance = np.tile(np.eye(2), (self.horizon, 1, 1))

        for iter in range(self.iterations):
            actions = self._sample_actions(mean, covariance)
            costs = np.zeros(self.num_samples)

            for i, action_seq in enumerate(actions):
                current_state = ego_state.copy()
                for t, action in enumerate(action_seq):
                    #print('Iteration: {}, Sample: {}, Time: {} with action {}'.format(iter, i, t, action))
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

                    next_state_bicyle = bicycle_model(current_state[:5].copy(), action, self.dt)
                    #print('Next state: {}, \n Next state bicycle: {}'.format([round(o,1) for o in next_state], [round(o,1) for o in next_state_bicyle]))
                    next_state[:5] = next_state_bicyle

                    costs[i] += self._cost(next_state, action, self.desired_trajectories, exo_states, exo_predictions, t)
                    current_state = next_state

            elites_indices = np.argsort(costs)[:self.num_elites]
            elites = actions[elites_indices]

            if (iter+1) % 50 == 0:
                print("Iteration {} with costs: {} - {}".format(iter, costs[elites_indices[0]], costs[elites_indices[1]]))

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

            return [center + np.matmul(rotation_matrix, corner) for corner in corners]

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

    def _simplified_check_collision(self, ego_state, exo_pos_at_t, exo_state_at_0):
        ego_radius = 0.5 * ego_state[6]  # half of the ego vehicle length
        exo_radius = 0.5 * exo_state_at_0[6]  # half of the exo vehicle length

        distance = np.linalg.norm(ego_state[:2] - exo_pos_at_t)
        min_distance = ego_radius + exo_radius

        return distance < min_distance

    
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
        #print("Inside cost: next state {} action {} desired trajectory {} current t "\
        #"exo states {} exo predictions {} current t {}".format(next_state, action, desired_trajectory, exo_states, exo_predictions, current_t))
        tracking_error = np.linalg.norm(next_state[0:2] - desired_trajectory[current_t])
        #print('Shape of desired trajectory: ', desired_trajectory.shape)
        tracking_cost = self.weights[0] * tracking_error

        # Calculate the yaw error cost
        # desired_yaw = np.arctan2(desired_trajectory[current_t][1] - next_state[1], desired_trajectory[current_t][0] - next_state[0])
        # yaw_error = (next_state[2] - desired_yaw) % (2 * np.pi)
        # yaw_error = np.minimum(yaw_error, 2 * np.pi - yaw_error)
        # yaw_cost = self.weights[1] * yaw_error

        # Calculate the speed encouragement cost
        speed_diff = (self.desired_speed - np.linalg.norm(next_state[3:5]))**2
        speed_cost = self.weights[2] * speed_diff

        control_effort_cost = yaw_cost + speed_cost
        # Calculate the control effort cost
        #control_effort_cost = self.weights[1] * (np.linalg.norm(next_state[3:5]) - self.desired_speed)**2
        #control_effort_cost += 5.0 * action[1]**2 # Penalize yaw rate

        # Calculate the collision cost
        collision_cost = 0
        for exo_id, exo_pred in exo_predictions.items():
            exo_state_at_0 = exo_states[0] # To know the initial position, heading, and bbox to find the bbox of future timestep
            exo_pred_at_t = exo_pred[current_t] # To know the future position
            if self._simplified_check_collision(next_state, exo_pred_at_t, exo_state_at_0):
                collision_cost = self.weights[3]
                break

        # Calculate the total cost
        total_cost = tracking_cost + control_effort_cost + collision_cost
        # if (current_t+1) % 10 == 0:
        #     print("Cost at time step {}: tracking  {} control {} collision {}, state {} total cost {}".format(current_t, tracking_cost, control_effort_cost, collision_cost,
        #                 [round(o,1) for o in next_state] ,total_cost))

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

    def update_trajectories(self, new_trajectories):
        '''
            Update the desired trajectories
            At every timestep, we need to update new trajectories.
            Because the desired trajectory is a function of time.
        '''
        self.desired_trajectories = new_trajectories

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = 0.03  # Default value is usually 0.1 seconds per frame
world.apply_settings(settings)
run_flag = True
def run_simulation(world, num_ticks, tick_interval):
    settings.synchronous_mode = True
    world.apply_settings(settings)

    while run_flag:
        world.tick()
        time.sleep(tick_interval)

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
    min_safe_points=100, rng=random.Random(90))

print("Path[0]: ", path[0], " position: ", get_position(path[0]))
# Creating carla waypoints

values = [get_position(path[i]) for i in range(len(path) - 1)]
print(values[0]) # A vector2D
draw_waypoints(values, world, color=carla.Color(r=0, g=255, b=0), life_time=200)

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

# Get the vehicle's physics control object
physics_control = vehicle.get_physics_control()
wheel_positions = [w.position / 100 for w in physics_control.wheels]

# Get the positions of the front and rear wheels
front_left_wheel = wheel_positions[0]
front_right_wheel = wheel_positions[1]
rear_left_wheel = wheel_positions[2]
rear_right_wheel = wheel_positions[3]
# print(physics_control.wheels[0].position.x)
# print(physics_control.wheels[0].position.y)
# print(front_left_wheel.x, front_left_wheel.y)
# print(front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel)
# Calculate the front and rear axle positions

front_left_wheel_np = np.array([front_left_wheel.x, front_left_wheel.y])
front_right_wheel_np = np.array([front_right_wheel.x, front_right_wheel.y])
rear_left_wheel_np = np.array([rear_left_wheel.x, rear_left_wheel.y])
rear_right_wheel_np = np.array([rear_right_wheel.x, rear_right_wheel.y])

front_axle_position = (front_left_wheel_np + front_right_wheel_np) / 2
rear_axle_position = (rear_left_wheel_np + rear_right_wheel_np) / 2

# Calculate the wheelbase (L)
L = np.linalg.norm(np.array(front_axle_position) - np.array(rear_axle_position))


# Set up the CrossEntropyMPC
horizon = 5
num_samples = 200
num_elites = 50
desired_trajectories = np.array([[val.x, val.y] for val in values])
dt = 3
iterations = 100
weights = np.array([50.0, 10.0, 10.0, 100.0])
mpc = CrossEntropyMPC(iterations, horizon, num_samples, num_elites, desired_trajectories, weights, 0.3, L)

import threading


# Main control loop
try:
    simulation_thread = threading.Thread(target=run_simulation, args=(world, 1, 0.3))
    simulation_thread.start()
    while True:
        start_time = time.time()
        
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
        print('Current state: ', vehicle_state)

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
        exo_vehicles = {}
        exo_predictions = {}

        # Optimize control inputs using CrossEntropyMPC
        optimal_action = mpc.optimize(vehicle_state, exo_vehicles, exo_predictions)
        print('Optimal action: ', optimal_action)
        # Apply control inputs to the ego vehicle
        control = carla.VehicleControl()
        control.throttle = max(0, min(1, optimal_action[0]))
        control.steer = max(-1, min(1, optimal_action[1]))
        vehicle.apply_control(control)

        # Update trajectories
        desired_trajectories = desired_trajectories.copy()[1:]
        mpc.update_trajectories(desired_trajectories)

        # Wait for dt seconds
        time.sleep(max(0, dt - (time.time() - start_time)))
        print("Time taken: ", time.time() - start_time)

finally:
    vehicle.destroy()
    run_flag = False

    simulation_thread.join()
    # Don't forget to reset the fixed_delta_seconds and synchronous_mode when you're done
    settings.synchronous_mode = False
    world.apply_settings(settings)


