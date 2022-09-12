import os
import fnmatch
import argparse
import numpy as np
import math
import random

cap = 10 

def collect_txt_files(rootpath, flag):
    txt_files = list([])
    for root, dirnames, filenames in os.walk(rootpath):

        if flag in root and ignore_flag not in root and 'debug' not in root:
            # print("subfolder %s found" % root)
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_files.append(os.path.join(root, filename))
    print("%d files found in %s" % (len(txt_files), rootpath))
    return txt_files


def filter_txt_files(root_path, txt_files):
    # container for files to be converted to h5 data
    filtered_files = list([])
    
    no_aa_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    for txtfile in txt_files:
        ok_flag = False
        no_aa = False
        with open(txtfile, 'r') as f:
            for line in reversed(list(f)):
                if 'Step {}'.format(cap + 1) in line or 'step {}'.format(cap + 1) in line:
                    ok_flag = True
                if 'No agent array messages received after' in line:
                    no_aa_count += 1 
                    no_aa = True 
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


def get_statistics(root_path, filtered_files):
    total_count = len(filtered_files)
    col_count = 0
    goal_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    eps_step = []
    goal_step = []
    ave_speeds = [] 
    dec_counts = []
    acc_counts = []
    mat_counts = []
    trav_dists= []
    for txtfile in filtered_files:
        #
        reach_goal_flag = False
        collision_flag = False
        cur_step = 0
        dec_count = 0
        acc_count = 0
        mat_count = 0
        speed = 0.0
        last_speed = 0.0
        ave_speed = 0.0
        dist = 0.0
        last_pos = None


        with open(txtfile, 'r') as f:
            data_pos = {}
            cur_step = 0

            for line in f:
                if 'executing step' in line:
                    line_1 = line.split('executing step ', 1)[1]
                    cur_step = int(line_1.split('=', 1)[0])
                elif 'Round 0 Step' in line:
                    line_1 = line.split('Round 0 Step ', 1)[1]
                    cur_step = int(line_1.split('-', 1)[0])
                elif 'goal reached at step' in line:
                    line_1 = line.split('goal reached at step ', 1)[1]
                    cur_step = int(line_1.split(' ', 1)[0])
                elif ("pomdp" in folder or "gamma" in folder or "rollout" in folder) and "car pos / heading / vel" in line: 
                    # = (149.52, 171.55) / 1.3881 / 0.50245
                    speed = float(line.split(' ')[12])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    if cur_step >= cap:
                        ave_speed += speed
                    pos = [pos_x, pos_y]

                    if last_pos:
                        dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                    last_pos = pos
                    if "gamma" in folder or 'pomdp' in folder or "rollout" in folder:
                        if speed< last_speed - 0.2:
                            dec_count += 1
                        last_speed = speed

                    if "av" in data_pos:
                        val = data_pos["av"]
                        val[cur_step] = pos
                    else:
                        data_pos["av"] = {cur_step: pos}

                elif ("pomdp" in folder or "gamma" in folder or "rollout" in folder) and "id / pos / speed / vel" in line:
                    agentid = int(line.split()[16].replace('(', '').replace(',', ''))
                    pos_x = float(line.split()[18].replace('(', '').replace(',', ''))
                    pos_y = float(line.split()[19].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]

                    if f"agent{agentid}" in data_pos:
                        val = data_pos[f"agent{agentid}"]
                        val[cur_step] = pos
                    else:
                        data_pos[f"agent{agentid}"] = {cur_step: pos}

                if 'goal reached' in line:
                    reach_goal_flag = True
                    break    
                if ('collision = 1' in line or 'INININ' in line or 'in real collision' in line) and reach_goal_flag == False:
                    collision_flag = True
                    col_count += 1
                    break

            if  cur_step >= 50:
                # 1. Get only agent's whose trajectory length bigger than 45 (we can pad something) and start at time 0
                considered_agents = {}
                for k, v in data_pos.items():
                    if ("agent" in k) and (0 in v.keys()):
                        considered_agents[k] = v

                # 2. We find the interested agent by sampling a list of nearest agents to the car and trajectory of nearest agent >= 50
                car_pos = np.array(list(data_pos["av"].values()))
                nearest_values = {}
                for k, v in considered_agents.items():
                    pos = np.array(list(v.values()))
                    if pos.shape[0] >= 50:
                        val = np.mean(np.sum((pos[0:50] - car_pos[0:50])**2, axis=-1))
                        nearest_values.setdefault(val, k)

                # Get at most 5 interested neareast agents
                sorted_nearest = sorted(nearest_values.items(), key = lambda x : x[0])
                random_interested_agent = random.choice(sorted_nearest[0:3])

                # 3. Building file



    print("%d filtered files found in %s" % (len(filtered_files), root_path))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--flag',
        type=str,
        default='test',
        help='Folder name to track')
    parser.add_argument(
        '--ignore',
        type=str,
        default='map_test',
        help='folder flag to ignore')
    parser.add_argument(
        '--folder',
        type=str,
        default='./',
        help='Subfolder to check')

    flag = parser.parse_args().flag
    folder = parser.parse_args().folder
    ignore_flag = parser.parse_args().ignore

    files = collect_txt_files(folder, flag)
    filtered = filter_txt_files(folder, files)
    get_statistics(folder, filtered)




