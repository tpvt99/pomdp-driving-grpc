import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys
import math
import matplotlib as mpl
import matplotlib.lines as mlines

cap = 20

def collect_txt_files(rootpath, flag, ignore_flag="nothing"):
    txt_files = list([])
    for root, dirnames, filenames in os.walk(rootpath):

        if flag in root and ignore_flag not in root and 'debug' not in root:
            # print("subfolder %s found" % root)
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_files.append(os.path.join(root, filename))
    print("%d files found in %s" % (len(txt_files), rootpath))
    return txt_files


# The function filters out that is good text files
def filter_txt_files(root_path, txt_files, cap=10):
    # container for files to be converted to h5 data
    filtered_files = list([])

    no_aa_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    for txtfile in txt_files:
        ok_flag = False
        no_aa = False
        with open(txtfile, 'r') as f:
            try:
                for line in reversed(list(f)):
                    if 'Step {}'.format(cap + 1) in line or 'step {}'.format(cap + 1) in line:
                        ok_flag = True
                    if 'No agent array messages received after' in line:
                        no_aa_count += 1
                        no_aa = True
            except:
                print(txtfile)
        if ok_flag == True:
            filtered_files.append(txtfile)
            # print("good file: ", txtfile)
        else:
            if no_aa:
                pass # print("no aa file: ", txtfile)
            else:
                pass # print("unused file: ", txtfile)
    print("NO agent array in {} files".format(no_aa_count))

    filtered_files.sort()
    print("%d filtered files found in %s" % (len(filtered_files), root_path))
    # print (filtered_files, start_file, end_file)
    #
    return filtered_files


# This function parse logging of DESPOT and return a dictionary of data
def parse_data(txt_file):
    action_list = {}
    ego_list = {}
    exos_list = {}
    coll_bool_list = {}
    ego_path_list = {}
    pred_car_list = {}
    pred_exo_list = {}
    trial_list = {}
    depth_list = {}
    expanded_nodes = {}
    total_nodes = {}
    gamma_prediction_time = []

    exo_count = 0
    start_recording = False
    start_recording_prediction = False

    with open(txt_file, 'r') as f:
        for line in f:
            if 'Round 0 Step' in line:
                line_split = line.split('Round 0 Step ', 1)[1]
                cur_step = int(line_split.split('-', 1)[0])
                start_recording = True
                start_recording_prediction = True
            if not start_recording:
                continue
            try:
                if "car pos / heading / vel" in line:  # ego_car info
                    speed = float(line.split(' ')[12])
                    heading = float(line.split(' ')[10])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    bb_x = float(line.split(' ')[15])
                    bb_y = float(line.split(' ')[16])

                    pos = [pos_x, pos_y]

                    agent_dict = {'pos': [pos_x, pos_y],
                                  'heading': heading,
                                  'speed': speed,
                                  'vel': (speed * math.cos(heading), speed * math.sin(heading)),
                                  'bb': (bb_x, bb_y)
                                  }
                    ego_list[cur_step] = agent_dict
                elif " pedestrians" in line:  # exo_car info start
                    exo_count = int(line.split(' ')[0])
                    exos_list[cur_step] = []
                elif "id / pos / speed / vel / intention / dist2car / infront" in line:  # exo line, info start from index 16
                    # agent 0: id / pos / speed / vel / intention / dist2car / infront =  54288 / (99.732, 462.65) / 1 / (-1.8831, 3.3379) / -1 / 9.4447 / 0 (mode) 1 (type) 0 (bb) 0.90993 2.1039 (cross) 1 (heading) 2.0874
                    line_split = line.split(' ')
                    agent_id = int(line_split[16 + 1])

                    pos_x = float(line_split[18 + 1].replace('(', '').replace(',', ''))
                    pos_y = float(line_split[19 + 1].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]
                    vel_x = float(line_split[23 + 1].replace('(', '').replace(',', ''))
                    vel_y = float(line_split[24 + 1].replace(')', '').replace(',', ''))
                    vel = [vel_x, vel_y]
                    bb_x = float(line_split[36 + 1])
                    bb_y = float(line_split[37 + 1])
                    heading = float(line_split[41 + 1])
                    agent_dict = {'id': agent_id,
                                  'pos': [pos_x, pos_y],
                                  'heading': heading,
                                  'vel': [vel_x, vel_y],
                                  'bb': (bb_x, bb_y)
                                  }

                    exos_list[cur_step].append(agent_dict)
                    assert (len(exos_list[cur_step]) <= exo_count)
                elif "Path: " in line:  # path info
                    # Path: 95.166 470.81 95.141 470.86 ...
                    line_split = line.split(' ')
                    path = []
                    for i in range(1, len(line_split) - 1, 2):
                        x = float(line_split[i])
                        y = float(line_split[i + 1])
                        path.append([x, y])
                    ego_path_list[cur_step] = path
                elif 'predicted_car_' in line and start_recording_prediction:
                    # predicted_car_0 378.632 470.888 5.541
                    # (x, y, heading in rad)
                    line_split = line.split(' ')
                    pred_step = int(line_split[0][14:])
                    x = float(line_split[1])
                    y = float(line_split[2])
                    heading = float(line_split[3])
                    agent_dict = {'pos': [x, y],
                                  'heading': heading,
                                  'bb': (10.0, 10.0)
                                  }
                    if pred_step == 0:
                        pred_car_list[cur_step] = []
                    pred_car_list[cur_step].append(agent_dict)

                elif 'predicted_agents_' in line and start_recording_prediction:
                    # predicted_agents_0 380.443 474.335 5.5686 0.383117 1.1751
                    # [(x, y, heading, bb_x, bb_y)]
                    line_split = line.split(' ')
                    if line_split[-1] == "" or line_split[-1] == "\n":
                        line_split = line_split[:-1]
                    pred_step = int(line_split[0][17:])
                    if pred_step == 0:
                        pred_exo_list[cur_step] = []
                    num_agents = (len(line_split) - 1) / 5
                    agent_list = []
                    for i in range(int(num_agents)):
                        start = 1 + i * 5
                        x = float(line_split[start])
                        y = float(line_split[start + 1])
                        heading = float(line_split[start + 2])
                        bb_x = float(line_split[start + 3])
                        bb_y = float(line_split[start + 4])
                        agent_dict = {'pos': [x, y],
                                      'heading': heading,
                                      'bb': (bb_x, bb_y)
                                      }
                        agent_list.append(agent_dict)
                    pred_exo_list[cur_step].append(agent_list)
                elif 'INFO: Executing action' in line:
                    line_split = line.split(' ')
                    steer = float(line_split[5].split('/')[0])
                    acc = float(line_split[5].split('/')[1])
                    speed = float(line_split[5].split('/')[2])
                    action_list[cur_step] = (steer, acc, speed)
                    # INFO: Executing action:22 steer/acc = 0/3
                elif "Trials: no. / max length" in line:
                    line_split = line.split(' ')
                    trial = int(line_split[6])
                    depth = int(line_split[8])
                    trial_list[cur_step] = trial
                    depth_list[cur_step] = depth
                if 'collision = 1' in line or 'INININ' in line or 'in real collision' in line:
                    coll_bool_list[cur_step] = 1

                if "# nodes: expanded" in line:
                    expanded, total, policy = line.split("=")[-1].split("/")
                    expanded_nodes[cur_step] = int(expanded)
                    total_nodes[cur_step] = int(total)

                if "[PredictAgents] Reach terminal state" in line: # if this is despot
                    pred_car_list.pop(cur_step, None)
                    pred_exo_list.pop(cur_step, None)
                    start_recording_prediction = False

                if "Prediction status:" in line: # If this is gamma
                    if "Success" not in line:
                        assert False, "This file is GAMMA prediction and it fail in prediction"
                if "Time taken" in line and "GAMMA" in line:
                    gamma_prediction_time.append(float(line.split(':')[-1]))

                if "Error in prediction" in line: # if this is gamma
                    assert False, "This file is in gamma prediction and it fail in prediction"

            except Exception as e:
                error_handler(e)
                assert False
                #pdb.set_trace()

    return action_list, ego_list, ego_path_list, exos_list, coll_bool_list, \
        pred_car_list, pred_exo_list, trial_list, depth_list, expanded_nodes, total_nodes, gamma_prediction_time


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    
def displacement_error(pred, gt):
    return np.linalg.norm(np.array(pred) - np.array(gt))

def displacement_error_2d(pred_list, gt_list):
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    return np.mean(np.sqrt(((pred_array-gt_array)**2).sum(1)))

def ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list, pred_len, obs_len=20):
    # Calculate ADE and FDE for each exo agent
    exo_ade = []
    exo_ade_obs20 = []
    exo_ade_closest = []
    for timestep in list(exos_list.keys()):
        
        if timestep not in pred_exo_list.keys() or timestep < 20:
            continue

        for exo_index, exo_agent in enumerate(exos_list[timestep]):
             
            agent_info = exo_agent
            agent_id = agent_info['id']   

            exo_pred_at_timestep = pred_exo_list[timestep]
            
            exo_obs_list = [] # exo observation
            exo_pred_list = [] # exo prediction
            exo_ggt_list = [] # exo ground truth
            
            for j in range(obs_len):
                all_agent_ids_at_obs_timestep = [agent['id'] for agent in exos_list[timestep-j]]
                if agent_id in all_agent_ids_at_obs_timestep:
                    exo_index_at_obs_timestep = all_agent_ids_at_obs_timestep.index(agent_id)
                    exo_past = exos_list[timestep-j][exo_index_at_obs_timestep]['pos']
                    exo_obs_list.append(exo_past)

            for j in range(pred_len):
                if (timestep+j+1) not in exos_list.keys():
                    break
                all_agent_ids_at_next_timestep = [agent['id'] for agent in exos_list[timestep+j+1]]
                # If agent is not in the next timestep, then there is no point in calculating the error
                if agent_id in all_agent_ids_at_next_timestep:
                    exo_index_at_next_timestep = all_agent_ids_at_next_timestep.index(agent_id)
                    exo_pred = exo_pred_at_timestep[j][exo_index]['pos']
                    exo_gt_next = exos_list[timestep+j+1][exo_index_at_next_timestep]['pos']
                    exo_pred_list.append(exo_pred)
                    exo_ggt_list.append(exo_gt_next)
            
            if len(exo_pred_list) == pred_len and len(exo_ggt_list) == pred_len:
                ade = displacement_error_2d(exo_pred_list, exo_ggt_list)
                exo_ade.append(ade)
                if len(exo_obs_list) == obs_len:
                    exo_ade_obs20.append(ade)
                if exo_index in [0,1,2]:
                    exo_ade_closest.append(ade)

    if len(exo_ade) == 0:
        return None, None, None, None, None

    # Calculate average ADE for each exo agent
    # We average all time steps for each agent. Then we average all agents.
    exo_ade_mean = np.mean(exo_ade)
    exo_ade_mean_obs20 = np.mean(exo_ade_obs20)
    exo_ade_mean_closest = np.mean(exo_ade_closest)

    return exo_ade_mean, exo_ade_mean_obs20, exo_ade_mean_closest, exo_ade, exo_ade_closest


from temporal_consistency_calculation import calculate_consistency
from driving_performance_safety import find_safety, find_safety_agent
from driving_performance_comfort import find_acceleration_and_jerk
from driving_performance_efficiency import efficiency_time_traveled, average_speed, path_tracking_error, distance_traveled


def get_dynamic_ade(ABSOLUTE_DIR, pred_len):

    prediction_performance = {
        'ade': [],
        'ade_obs20': [],
        'ade_closest': [],
        'std': [],
        'std_closest': [],
        'temp_consistency': [],
        'temp_consistency_closest': []
    }

    driving_performance = {
        'safety': {
            'collision_rate': [],
            'near_miss_rate': [],
            'near_distance_rate': []
        },
        'comfort': {
            'jerk': [],
            'lateral_acceleration': [],
            'acceleration': []
        },
        'efficiency': {
            'avg_speed': [],
            'tracking_error': [],
            'efficiency_time': [],
            'distance_traveled': []
        },
    }

    for root, subdirs, files in os.walk(ABSOLUTE_DIR):
        if len(files) > 0:
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    print(f"\nProcessing {file_path}")
                    try:
                        action_list, ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list, \
                                trial_list, depth_list, expanded_nodes, total_nodes, gamma_time = parse_data(file_path)
                    except:
                        continue

                    # The number of steps are too small which can affect ADE/FDE, thus we ignore these files
                    if len(ego_list) <= 80:
                        continue

                    # Prediction performance
                    try:
                        exo_ade, exo_ade_obs20, exo_ade_closest, exo_ade_distribution, exo_ade_closest_distribution = \
                                ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list, pred_len)
                        # import pdb;pdb.set_trace()
                        if exo_ade is None:
                            continue
                    except Exception as e:
                        print(e)
                        continue
                    tmp_consistency, temp_consistency_closest = calculate_consistency(exos_list, pred_exo_list, 5)
                    prediction_performance['ade'].append(exo_ade)
                    prediction_performance['ade_obs20'].append(exo_ade_obs20)
                    prediction_performance['ade_closest'].append(exo_ade_closest)
                    prediction_performance['std'].append(np.std(exo_ade_distribution))
                    prediction_performance['std_closest'].append(np.std(exo_ade_closest_distribution))
                    prediction_performance['temp_consistency'].append(tmp_consistency)
                    prediction_performance['temp_consistency_closest'].append(temp_consistency_closest)
                    print(f"Dynamic ADE {exo_ade:.2f}, Dynamic ADE OBS20  {exo_ade_obs20:.2f}, Dynamic ADE CLOSEST {exo_ade_closest:.2f}", end=" ")

                    # Driving performance - safety
                    collision_rate, near_miss_rate, near_distance_rate = find_safety(ego_list, exos_list)
                    # collision_rate, near_miss_rate = find_safety_agent(ego_list, exos_list)
                    driving_performance['safety']['collision_rate'].append(collision_rate)
                    driving_performance['safety']['near_miss_rate'].append(near_miss_rate)
                    driving_performance['safety']['near_distance_rate'].append(near_distance_rate)
                    print(f"collision rate {collision_rate:.2f}, near_miss_rate {near_miss_rate:.2f}, near_distance_rate {near_distance_rate:.2f}", end=" ")
                    # print(f"collision rate agent {collision_rate2:.2f}, near_miss_rate_agent{near_miss_rate2:.2f}", end=" ")

                    # Driving performance - comfort
                    jerk, lateral_acceleration, acceleration = find_acceleration_and_jerk(ego_list)
                    driving_performance['comfort']['jerk'].append(jerk)
                    driving_performance['comfort']['lateral_acceleration'].append(lateral_acceleration)
                    driving_performance['comfort']['acceleration'].append(acceleration)

                    print(f"jerk {jerk:.2f}, lat-acc {lateral_acceleration:.2f}", end=" ")

                    # Driving performance - efficiency
                    avg_speed = average_speed(ego_list)
                    tracking_error = path_tracking_error(ego_list, ego_path_list)
                    efficiency_time = efficiency_time_traveled(ego_list, ego_path_list)
                    distance_travel = distance_traveled(ego_list)
                    driving_performance['efficiency']['avg_speed'].append(avg_speed)
                    driving_performance['efficiency']['tracking_error'].append(tracking_error)
                    driving_performance['efficiency']['efficiency_time'].append(efficiency_time)
                    driving_performance['efficiency']['distance_traveled'].append(distance_travel)

                    print(f"avg_speed {avg_speed:.2f}, track-err {tracking_error:.2f} eff-time {efficiency_time:.2f}", end=" ")    

    return prediction_performance, driving_performance