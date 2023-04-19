import numpy as np
from typing import List
from shapely.geometry import Polygon
import math

def get_corners(agent, buffer):
    width, length = agent['bb']
    x, y = agent['pos']
    heading = agent['heading']

    dx = length / 2
    dy = width / 2

    corners = [
        [x - dx - buffer, y - dy],
        [x + dx + buffer, y - dy],
        [x + dx + buffer, y + dy],
        [x - dx - buffer, y + dy],
    ]

    # Rotate corners based on heading
    rotated_corners = []
    for corner in corners:
        x_diff = corner[0] - x
        y_diff = corner[1] - y
        new_x = x + (x_diff * np.cos(heading) + y_diff * np.sin(heading))
        new_y = y + (x_diff * np.sin(heading) - y_diff * np.cos(heading))
        rotated_corners.append([new_x, new_y])

    return rotated_corners

### ------------------ Safety 1 - Collision rate --------------------------

def check_collision_by_considering_headings(ego, exo, buffer=0):
    ego_corners = get_corners(ego,buffer)
    exo_corners = get_corners(exo,buffer)
    
    ego_polygon = Polygon(ego_corners)
    exo_polygon = Polygon(exo_corners)

    return ego_polygon.intersects(exo_polygon)

def check_collision_by_distance(ego, exo):
    x, y = ego['pos']
    x_exo, y_exo = exo['pos']
    return np.sqrt((x-x_exo)**2 + (y-y_exo)**2)

def find_safety_agent(ego_dict, exos_dict):
    collision_count = 0
    near_collision_count = 0

    near_threshold = 1
    total_time_steps = len(ego_dict)
    collided_exo_ids = set()
    near_collided_exo_ids = set()

    # Iterate through each time step in the episode
    for time_step in list(ego_dict.keys()):
        # We are not sure time_step is in exos_dict so we check
        if time_step in exos_dict.keys():
            ego_agent = ego_dict[time_step]
            exo_agents = exos_dict[time_step]
            # Chck for collisions with exo cars at the current time step
            for exo_agent in exo_agents:
                if check_collision_by_considering_headings(ego_agent, exo_agent, buffer=0) and exo_agent['id'] not in collided_exo_ids:
                    collision_count += 1
                    collided_exo_ids.add(exo_agent['id'])
                    break
                if check_collision_by_considering_headings(ego_agent, exo_agent, buffer=near_threshold) and exo_agent['id'] not in near_collided_exo_ids:
                    near_collision_count += 1
                    near_collided_exo_ids.add(exo_agent['id'])
                    break

    # Calculate the collision rate
    collision_rate = collision_count / total_time_steps
    near_collision_rate = near_collision_count / total_time_steps
    return collision_rate, near_collision_rate

def find_safety(ego_dict, exos_dict):
    
    near_threshold = 1.0
    near_distance = 2.0
    
    collision_count = 0
    near_collision_count = 0
    near_distance_count = 0

    total_time_steps = len(ego_dict)

    # Iterate through each time step in the episode
    for time_step in list(ego_dict.keys()):
        # We are not sure time_step is in exos_dict so we check
        if time_step in exos_dict.keys():
            ego_agent = ego_dict[time_step]
            exo_agents = exos_dict[time_step]
            # Chck for collisions with exo cars at the current time step
            for exo_agent in exo_agents:
                if check_collision_by_considering_headings(ego_agent, exo_agent, buffer=-0.5):
                    collision_count += 1
                    break
                if check_collision_by_considering_headings(ego_agent, exo_agent, buffer=near_threshold):
                    near_collision_count += 1
                    break
                if check_collision_by_distance(ego_agent, exo_agent):
                    near_distance_count += 1
                    break
    # Calculate the collision rate
    collision_rate = collision_count / total_time_steps
    near_collision_rate = near_collision_count / total_time_steps
    near_distance_rate = near_distance_count / total_time_steps
    return collision_rate, near_collision_rate, near_distance_rate