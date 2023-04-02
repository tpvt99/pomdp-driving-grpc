import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys
import math
import matplotlib as mpl
import matplotlib.lines as mlines

cap = 20

# The function file all text in a folder and its subfolders
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

    with open(txt_file, 'r') as f:
        for line in f:
            if 'Round 0 Step' in line:
                line_split = line.split('Round 0 Step ', 1)[1]
                cur_step = int(line_split.split('-', 1)[0])
                start_recording = True
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
                                  'bb': (bb_x * 2, bb_y * 2)
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
                elif 'predicted_car_' in line:
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

                elif 'predicted_agents_' in line:
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
                                      'bb': (bb_x * 2, bb_y * 2)
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
                    start_recording = False

                if "Prediction status:" in line: # If this is gamma
                    if "Success" not in line:
                        assert False, "This file is GAMMA prediction and it fail in prediction"
                if "Time taken" in line and "GAMMA" in line:
                    gamma_prediction_time.append(float(line.split(':')[-1]))

            except Exception as e:
                error_handler(e)
                #assert False
                pdb.set_trace()

    return action_list, ego_list, ego_path_list, exos_list, coll_bool_list, \
        pred_car_list, pred_exo_list, trial_list, depth_list, expanded_nodes, total_nodes, gamma_prediction_time


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    
def displacement_error(pred, gt):
    return np.linalg.norm(np.array(pred) - np.array(gt))

def ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list):
    ego_ade = []
    ego_fde = []
    ego_obs_list = {} # exo observation

    # Calculate ADE and FDE for ego agent
    for i, timestep in enumerate(pred_car_list):

        # Skip the first 20 timesteps because the prediction is not accurate
        if timestep < 30:
            continue

        errors = []
        num_pred = len(pred_car_list[timestep])
        ego_obs_list[timestep] = []
        
        for j in range(num_pred):
            if (timestep + j+1) in ego_list.keys():
                ego_pred = pred_car_list[timestep][j]['pos']
                ego_gt_next = ego_list[timestep+j+1]['pos'] # +1 because the prediction is for the next timestep
                error = displacement_error(ego_pred, ego_gt_next)
                errors.append(error)
                if j < 20 and (timestep +j - 20) in ego_list.keys():
                    ego_obs_list[timestep].append(ego_list[timestep + j - 20]['pos'])
        if len(errors) > 0:
            ego_ade.append(np.mean(errors))
            ego_fde.append(errors[-1])


    # Calculate ADE and FDE for each exo agent
    exo_ade = {}
    exo_fde = {}

    distributions = []

    for timestep in list(exos_list.keys()):

        # Skip the first 20 timesteps because the prediction is not accurate
        if timestep < 30:
            continue

        for agent_index, exo_agent in enumerate(exos_list[timestep]):

            # if agent_index != 0:
            #     continue

            agent_info = exo_agent
            agent_id = agent_info['id']

            if timestep not in pred_exo_list.keys():
                continue

            num_pred = len(pred_exo_list[timestep])
            exo_pred_at_timestep = pred_exo_list[timestep] # A list of 30 predictions. Each prediction is a list of agents
            
            exo_pred_list = [] # exo prediction
            exo_ggt_list = [] # exo ground truth
            exo_obs_list = [] # exo observation
            errors = []
            
            for j in range(num_pred):
                if (timestep + j+1) not in exos_list.keys():
                    break
                all_agent_ids_at_next_timestep = [agent['id'] for agent in exos_list[timestep+j+1]]
                all_agent_ids_at_prev_20_timestep = [agent['id'] for agent in exos_list[timestep+j-20]]
                
                # If agent is not in the next timestep, then there is no point in calculating the error
                if agent_id not in all_agent_ids_at_next_timestep:
                    break
                
                # If agent not in prev 20 timestep, then it not have enough history to calculate the error
                if agent_id not in all_agent_ids_at_prev_20_timestep:
                    break
                
                if j < 20:
                    exo_obs_list.append(exos_list[timestep+j-20][all_agent_ids_at_prev_20_timestep.index(agent_id)]['pos'])

                if agent_id in all_agent_ids_at_next_timestep:
                    agent_index_at_next_timestep = all_agent_ids_at_next_timestep.index(agent_id)
                    exo_pred = exo_pred_at_timestep[j][agent_index]['pos']
                    exo_gt_next = exos_list[timestep+j+1][agent_index_at_next_timestep]['pos']
                    exo_pred_list.append(exo_pred)
                    exo_ggt_list.append(exo_gt_next)
                    error = displacement_error(exo_pred, exo_gt_next)
                    errors.append(error)
            
            #if len(xxx) >= 10 and (np.abs(np.diff(np.array(xxx), axis=0)).sum() <= 0.1 or np.abs(np.diff(np.array(ooo), axis=0)).sum()) <= 0.1:
            #    assert False, f"xxx is {xxx}, agent_id is {agent_id} at timestep {timestep}"
            #if np.abs(np.diff(np.array(exo_obs_list), axis=0)).sum() <= 0.5:
            #     #assert False, f"error is {errors}, agent_id is {agent_id} at timestep {timestep} \n xxx is {xxx} \n ggt is {ggt} \n {ooo}"
                #print(f"agent id {agent_id} ggt: {exo_ggt_list} \n pred: {exo_pred_list} \n obs: {exo_obs_list}")
            #    continue
            #print(f"Error is {np.mean(errors)} at timestep {timestep} for agent {agent_id}")
            #if np.mean(errors) > 15:
            #    print(f"time {timestep} agent id {agent_id} ggt: {exo_ggt_list} \n pred: {exo_pred_list} \n obs: {exo_obs_list}")
            #    print(f"Error is {np.mean(errors)} at timestep {timestep} for agent {agent_id}")
            #    assert np.isclose(np.mean(np.linalg.norm(np.array(exo_pred_list) - np.array(exo_ggt_list), axis=1)), np.mean(errors))
            #    assert False, f"agent id {agent_id} at timestep {timestep} \n ego: {ego_obs_list[timestep]} \n ggt: {exo_ggt_list} \n pred: {exo_pred_list} \n obs: {exo_obs_list}"
            
            if len(errors) > 0:
                if agent_id not in exo_ade:
                    exo_ade[agent_id] = []
                    exo_fde[agent_id] = []
                exo_ade[agent_id].append(np.mean(errors))
                exo_fde[agent_id].append(errors[-1])

                distributions.append(float(np.mean(errors)))

    if len(exo_ade) == 0:
        return None, None, None, None, None

    # Calculate average ADE for each exo agent
    # We average all time steps for each agent. Then we average all agents.
    exo_ade = np.mean([np.mean(exo_ade[agent_id]) for agent_id in exo_ade.keys()])
    exo_fde = np.mean([np.mean(exo_fde[agent_id]) for agent_id in exo_fde.keys()])

    return np.mean(ego_ade), np.mean(ego_fde), exo_ade, exo_fde, distributions
