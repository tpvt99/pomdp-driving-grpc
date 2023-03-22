import carla
import numpy as np
import cvxpy as cp

# MPC parameters
N = 10  # prediction horizon
dt = 0.1  # time step (s)
target_speed = 10.0  # target speed (m/s)
max_accel = 2.0  # maximum acceleration (m/s^2)
max_steer = np.radians(30)  # maximum steering angle (rad)
safe_distance = 5.0  # safe distance (m)

def mpc_control(vehicle, target_trajectory, other_vehicles=[]):
    # Get the current state of the vehicle
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    vehicle_velocity = vehicle.get_velocity()

    # Create the optimization variables
    u = cp.Variable((2, N))  # control inputs (acceleration, steering angle)
    x = cp.Variable((4, N + 1))  # state variables (x, y, yaw, speed)

    # Set the initial state
    x_init = np.array([
        vehicle_location.x,
        vehicle_location.y,
        np.radians(vehicle_rotation.yaw),
        np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
    ])
    constraints = [x[:, 0] == x_init]

    # Set the MPC constraints and cost function
    cost = 0
    for t in range(N):
        # Cost function
        cost += cp.sum_squares(x[:2, t] - target_trajectory[:2, t])  # minimize distance to target trajectory
        cost += cp.sum_squares(x[3, t] - target_speed)  # minimize deviation from target speed

        # System dynamics
        x_t_next = x[:, t] + dt * np.array([
            x[3, t] * cp.cos(x[2, t]),
            x[3, t] * cp.sin(x[2, t]),
            x[3, t] * cp.tan(u[1, t]),
            u[0, t]
        ])
        constraints += [x[:, t + 1] == x_t_next]

        # Control input constraints
        constraints += [cp.abs(u[0, t]) <= max_accel]  # acceleration
        constraints += [cp.abs(u[1, t]) <= max_steer]  # steering angle


        # Collision avoidance constraints
        for other_vehicle in other_vehicles:
            other_vehicle_transform = other_vehicle.get_transform()
            other_vehicle_location = other_vehicle_transform.location
            other_vehicle_velocity = other_vehicle.get_velocity()
            other_vehicle_speed = np.sqrt(other_vehicle_velocity.x**2 + other_vehicle_velocity.y**2)

            # Compute the distance between the ego vehicle and the other vehicle
            distance = np.sqrt((other_vehicle_location.x - vehicle_location.x)**2 + (other_vehicle_location.y - vehicle_location.y)**2)

            # Compute the time to collision
            if other_vehicle_speed > 0:
                time_to_collision = distance / other_vehicle_speed
            else:
                time_to_collision = np.inf

            # Add a collision avoidance constraint if the time to collision is less than the prediction horizon
            if time_to_collision < N * dt:
                # Compute the position of the other vehicle at each time step
                other_vehicle_trajectory = np.zeros((2, N + 1))
                for i in range(N + 1):
                    other_vehicle_trajectory[0, i] = other_vehicle_location.x + i * other_vehicle_speed * dt
                    other_vehicle_trajectory[1, i] = other_vehicle_location.y

                # Add a collision avoidance constraint
                constraints += [cp.sum_squares(x[:2, t] - other_vehicle_trajectory[:2, t]) >= safe_distance**2]


    # Solve the MPC problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    # Return the optimal control inputs
    return u[:, 0].value

def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    # Get the world and the vehicle
    world = client.get_world()
    vehicle = world.get_actors().filter('vehicle.*')[0]  # assuming there is already a vehicle in the simulation

    # Main control loop
    try:
        while True:
            # Generate a target trajectory for the vehicle to follow
            # Replace this with your desired trajectory generation method
            target_trajectory = np.zeros((2, N + 1))


            for i in range(N + 1):
                target_trajectory[0, i] = vehicle.get_location().x + i * target_speed * dt
                target_trajectory[1, i] = vehicle.get_location().y

            # Get other vehicles in the environment
            other_vehicles = world.get_actors().filter('vehicle.*')
            other_vehicles = [v for v in other_vehicles if v.id != vehicle.id]

            # Call the MPC controller to compute the control inputs
            accel, steer = mpc_control(vehicle, target_trajectory, other_vehicles)

            # Apply the control inputs to the vehicle
            control = carla.VehicleControl(throttle=max(accel, 0), brake=-min(accel, 0), steer=float(steer))
            vehicle.apply_control(control)

    except KeyboardInterrupt:
        print("MPC control loop stopped by user.")
        pass

if __name__ == "__main__":
    main()
