import os
import sys
import pdb
import math
import pickle
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from common_performance_stats import collect_txt_files, filter_txt_files, parse_data, ade_fde
from driving_performance_safety import find_safety
from driving_performance_comfort import find_acceleration_and_jerk
from driving_performance_efficiency import efficiency_time_traveled, average_speed, path_tracking_error, distance_traveled
import pandas as pd
from temporal_consistency_calculation import calculate_consistency
pred_len = 30

def get_dynamic_ade(ABSOLUTE_DIR):

    prediction_performance = {
        'ade': [],
        'ade_obs20': [],
        'ade_obs20_closest': [],
        'distribution': [],
        'std': [],
        'temp_consistency': []
    }

    driving_performance = {
        'safety': {
            'collision_rate': [],
            'near_miss_rate': [],
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
            'distance_traveled': [],
        },
    }

    for root, subdirs, files in os.walk(ABSOLUTE_DIR):
        if len(files) > 0:
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    print(f"\nProcessing {file_path}")
                    try:
                        action_list, ego_list, ego_path_list, exos_list, coll_bool_list, \
                        pred_car_list, pred_exo_list, trial_list, depth_list, expanded_nodes, total_nodes, gamma_time = parse_data(file_path)
                    except:
                        continue

                    # The number of steps are too small which can affect ADE/FDE, thus we ignore these files
                    if len(ego_list) <= 20:
                        continue

                    # Prediction performance
                    try:
                        exo_ade, exo_ade_obs20, exo_ade_closest, exo_ade_distribution, exo_ade_closest_distribution = \
                                ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list, pred_len)
                        if exo_ade is None:
                            continue
                    except Exception as e:
                        print(e)
                        continue
                    prediction_performance['ade'].append(exo_ade)
                    prediction_performance['ade_obs20'].append(exo_ade_obs20)
                    prediction_performance['ade_obs20_closest'].append(exo_ade_closest)
                    prediction_performance['distribution'].append(exo_ade_distribution)
                    prediction_performance['std'].append(np.std(exo_ade_distribution))
                    print(f"Dynamic ADE {exo_ade:.2f}, Dynamic ADE OBS20  {exo_ade_obs20:.2f}, Dynamic ADE CLOSEST {exo_ade_closest:.2f}", end=" ")

    return prediction_performance

if __name__ == "__main__":

    methods = ['cv','ca','knndefault','knnsocial','hivt','lanegcn','lstmdefault','lstmsocial']
    performance_summit = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance = np.zeros(len(methods))
    performance_obs20 = np.zeros(len(methods))

    directories_map= {
        #'lstmsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmsocial5Hz',
        'cv': '/home/phong/driving_data/official/despot_planner/smu_server/cv1Hz',
        #'ca': '/home/phong/driving_data/official/despot_planner/same_computation/ca2Hz',
        #'lanegcn': '/home/phong/driving_data/official/despot_planner/same_computation/lanegcn02Hz',
        #'lanegcn': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lanegcn/'
        #'lstmdefault': '/home/phong/driving_data/official/despot_planner/same_computation/lstmdefault05Hz'
        # 'knndefault': '/home/phong/driving_data/official/despot_planner/same_computation/knndefault01Hz'
    }

    prediction_performance = {}

    for key in directories_map.keys():
        print(f"Processing {key}")
        prediction_performance = get_dynamic_ade(directories_map[key])
        prediction_performance[key] = prediction_performance

    for key in directories_map.keys():
        idx = methods.index(key)
        performance[idx] = np.nanmean(prediction_performance[key]['ade'])
        performance_obs20[idx] = np.nanmean(prediction_performance[key]['ade_obs20'])
        print(f"Mean ADE for {key} is {np.nanmean(prediction_performance[key]['ade']):.2f}")

    ## plot static ade vs dynamic ade ##
    offsets=0.25
    position = np.arange(len(methods)) + np.arange(len(methods))//4

    ## Reason 1: close-loop effect ~ Performance better because agent avoid others ##
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.bar(position-offsets, performance_summit, width=offsets, color='red', align='center', label='Summit_dataset')
    ax.bar(position, performance_obs20, width=offsets, color='blue', align='center', label='Checked method')
    ax.set_xticks(position)
    ax.set_xticklabels(methods)
    ax.set_title('Comparison of Static_ADE and Dynamic_ADE_obs20')
    ax.set_ylabel('ADE')
    ax.legend()
    plt.show()
    plt.savefig("1_Static_vs_Dynamic_ADE_obs20_checked.png")

    ## Comprehensive Comparison ##
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.bar(position-offsets, performance_summit, width=offsets, color='red', align='center', label='Summit_dataset')
    ax.bar(position, performance, width=offsets, color='#929591', align='center', label='Checked method')
    ax.set_xticks(position)
    ax.set_xticklabels(methods)
    ax.set_title('Comparison of Static_ADE and Dynamic_ADE')
    ax.set_ylabel('ADE')
    ax.legend()
    plt.show()
    plt.savefig("1_Static_vs_Dynamic_ADE_checked.png")
    print("Comprehensive Comparison Done")