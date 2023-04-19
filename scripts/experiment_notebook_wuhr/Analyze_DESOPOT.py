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
from driving_performance_safety import find_safety, find_safety_agent
from driving_performance_comfort import find_acceleration_and_jerk
from driving_performance_efficiency import efficiency_time_traveled, average_speed, path_tracking_error, distance_traveled
import pandas as pd
from temporal_consistency_calculation import calculate_consistency
import statsmodels.api as sm
import numpy as np

pred_len = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")

    parser.add_argument('--mode', help='Generate file or only plot the relation', required=True)

    args = parser.parse_args()

    return args

def get_dynamic_ade(ABSOLUTE_DIR):

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
                        if exo_ade is None:
                            continue
                    except Exception as e:
                        print(e)
                        continue
                    tmp_consistency, temp_consistency_closest = calculate_consistency(exos_list, pred_exo_list, pred_len)
                    prediction_performance['ade'].append(exo_ade)
                    prediction_performance['ade_obs20'].append(exo_ade_obs20)
                    prediction_performance['ade_closest'].append(exo_ade_closest)
                    prediction_performance['std'].append(np.std(exo_ade_distribution))
                    prediction_performance['std_closest'].append(np.std(exo_ade_closest_distribution))
                    prediction_performance['temp_consistency'].append(tmp_consistency)
                    prediction_performance['temp_consistency_closest'].append(temp_consistency_closest)
                    print(f"Dynamic ADE {exo_ade:.2f}, Dynamic ADE OBS20  {exo_ade_obs20:.2f}, Dynamic ADE CLOSEST {exo_ade_closest:.2f}", end=" ")

                    # Driving performance - safety
                    collision_rate, near_miss_rate = find_safety(ego_list, exos_list)
                    # collision_rate, near_miss_rate = find_safety_agent(ego_list, exos_list)
                    driving_performance['safety']['collision_rate'].append(collision_rate)
                    driving_performance['safety']['near_miss_rate'].append(near_miss_rate)

                    print(f"collision rate {collision_rate:.2f}, near_miss_rate {near_miss_rate:.2f}", end=" ")
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

def save_dict_to_file(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dictionary, f)

# Load the dictionary from a file
def load_dict_from_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def remove_outliers_iqr(X, y1, y2, y3, multiplier=1.5):
    """
    Remove outliers using the interquartile range (IQR) method.
    
    Args:
        X (list): Independent variable.
        y (list): Dependent variable.
        multiplier (float): Multiplier to determine the threshold for outlier detection.
            Default is 1.5, which is a common value used in practice.
    
    Returns:
        X_clean (list): Independent variable with outliers removed.
        y_clean (list): Dependent variable with outliers removed.
    """
    # Convert X and y lists to numpy arrays
    X = np.array(X)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    
    # Calculate the IQR for X and y
    Q1_X = np.percentile(X, 25)
    Q3_X = np.percentile(X, 75)
    IQR_X = Q3_X - Q1_X

    Q1_y1 = np.percentile(y1, 25)
    Q3_y1 = np.percentile(y1, 75)
    IQR_y1 = Q3_y1 - Q1_y1

    Q1_y2 = np.percentile(y2, 25)
    Q3_y2 = np.percentile(y2, 75)
    IQR_y2 = Q3_y2 - Q1_y2

    Q1_y3 = np.percentile(y3, 25)
    Q3_y3 = np.percentile(y3, 75)
    IQR_y3 = Q3_y3 - Q1_y3

    # Define the upper and lower bounds for outlier detection
    upper_bound_X = Q3_X + multiplier * IQR_X
    lower_bound_X = Q1_X - multiplier * IQR_X

    upper_bound_y1 = Q3_y1 + multiplier * IQR_y1
    lower_bound_y1 = Q1_y1 - multiplier * IQR_y1

    upper_bound_y2 = Q3_y2 + multiplier * IQR_y2
    lower_bound_y2 = Q1_y2 - multiplier * IQR_y2

    upper_bound_y3 = Q3_y3 + multiplier * IQR_y3
    lower_bound_y3 = Q1_y3 - multiplier * IQR_y3

    # Filter the data to keep only the values within the bounds
    X_clean = X[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) &
                (y2 >= lower_bound_y2) & (y2 <= upper_bound_y2) &
                (y3 >= lower_bound_y3) & (y3 <= upper_bound_y3)]
    y1_clean = y1[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) &
                (y2 >= lower_bound_y2) & (y2 <= upper_bound_y2) &
                (y3 >= lower_bound_y3) & (y3 <= upper_bound_y3)]
    y2_clean = y2[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) &
                (y2 >= lower_bound_y2) & (y2 <= upper_bound_y2) &
                (y3 >= lower_bound_y3) & (y3 <= upper_bound_y3)]
    y3_clean = y3[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) &
                (y2 >= lower_bound_y2) & (y2 <= upper_bound_y2) &
                (y3 >= lower_bound_y3) & (y3 <= upper_bound_y3)]

    return X_clean, y1_clean, y2_clean, y3_clean

def cut(prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data, number):
    return prediction_data[:number], safety_data[:number], comfort_data[:number], efficiency_data[:number], driving_performance_data[:number]

def plot_method(method, prediction_performance, driving_performance, max_min_method, row, axes, prediction_metric):

    # Function to compute the weighted average
    def normalize(arr, max_val = 1.0, min_val = 0.0):
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        
        return (arr - np.array(min_val)) / (np.array(max_val) - np.array(min_val)+1e-6)

    def weighted_average(data, weights, directions, max_min_list):
        norm_data = [normalize(arr, max_min[0], max_min[1]) if direction == 'lower'
                        else 1 - normalize(arr, max_min[0], max_min[1]) for arr, direction, max_min in zip(data, directions, max_min_list)]
        return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)
    
    def weighted_average_no_normalize(data, weights, directions, max_min_list):
        norm_data = [normalize(arr, 1, 0) for arr, direction, max_min in zip(data, directions, max_min_list)]
        return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)
    
    # Weights for each metric within a category
    safety_weights = {
        'collision_rate': 1.0,
        'near_miss_rate': 0.0,
    }
    safety_directions = {
        'collision_rate': 'lower',
        'near_miss_rate': 'lower',
    }
    comfort_weights = {
        'jerk': 0.5,
        'lateral_acceleration': 0.0,
        'acceleration': 0.5
    }
    comfort_directions = {
        'jerk': 'lower',
        'lateral_acceleration': 'lower',
        'acceleration': 'lower'
    }
    efficiency_weights = {
        'avg_speed': 1.0,
        'tracking_error': 0.0,
        'efficiency_time': 0.0,
        'distance_traveled': 0.0, 
    }
    efficiency_directions = {
        'avg_speed': 'higher',
        'tracking_error': 'lower',
        'efficiency_time': 'lower',
        'distance_traveled': 'higher'
    }

    color = 'red'
    
    # Compute the weighted averages for each category
    max_min_safety = [tuple(max_min_method['safety'][key].values()) for key in max_min_method['safety'].keys()]
    safety_data = weighted_average(
        [driving_performance[method]['safety']['collision_rate'],
        driving_performance[method]['safety']['near_miss_rate']],
        list(safety_weights.values()), list(safety_directions.values()),
        max_min_safety)

    max_min_comfort = [tuple(max_min_method['comfort'][key].values()) for key in max_min_method['comfort'].keys()]
    comfort_data = weighted_average(
        [driving_performance[method]['comfort']['jerk'],
        driving_performance[method]['comfort']['lateral_acceleration'],
        driving_performance[method]['comfort']['acceleration']],
        list(comfort_weights.values()), list(comfort_directions.values()), max_min_comfort)

    max_min_efficiency = [tuple(max_min_method['efficiency'][key].values()) for key in max_min_method['efficiency'].keys()]
    print(max_min_efficiency)
    efficiency_data = weighted_average(
        [driving_performance[method]['efficiency']['avg_speed'],
        driving_performance[method]['efficiency']['tracking_error'],
        driving_performance[method]['efficiency']['efficiency_time'],
        driving_performance[method]['efficiency']['distance_traveled']],
        list(efficiency_weights.values()), list(efficiency_directions.values()), max_min_efficiency)
    

    # max_values = [max(prediction_performance[name][prediction_metric]) for name in prediction_performance.keys()]
    # min_values = [min(prediction_performance[name][prediction_metric]) for name in prediction_performance.keys()]
    # prediction_data = normalize(np.array(prediction_performance[method][prediction_metric]), max(max_values), min(min_values))
    # print(f"current prediction metric is {prediction_metric}, max is {max(max_values)}, min is {min(min_values)}")
    prediction_data = np.array(prediction_performance[method][prediction_metric])
    
    prediction_data, safety_data, comfort_data, efficiency_data = remove_outliers_iqr(prediction_data, safety_data, comfort_data, efficiency_data, multiplier=1.5)

    driving_performance_data = (efficiency_data+safety_data+comfort_data)/3

    # prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data = cut(prediction_data, \
    #                 safety_data, comfort_data, efficiency_data, driving_performance_data, 10)
    # print(f"max min safety: {max_min_safety}")
    # print(f"max min comfort: {max_min_comfort}")
    # print(f"max min efficiency: {max_min_efficiency}")
    # Add scatter plots for each method
    # Add scatter plots for each method
    # figure1 = plt.scatter(prediction_data, comfort_data)
    # plt.xlabel(f"{prediction_metric}")
    # plt.savefig(f"try_{method}_{prediction_metric}.png")
    # plt.clf()

    axes[0].scatter(x=prediction_data, y=comfort_data, c='red', marker='o')
    axes[1].scatter(x=prediction_data, y=safety_data, c='red', marker='o')
    axes[2].scatter(x=prediction_data, y=efficiency_data, c='red', marker='o')
    axes[3].scatter(x=prediction_data, y=driving_performance_data, c='red', marker='o')

    return np.mean(driving_performance_data), prediction_data, safety_data, comfort_data, efficiency_data
    

def scatter_plot_multi_pred_3_metric(prediction_performance, driving_performance, methods_to_plot, prediction_metric, x_limit):  

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    def find_max_min(dpm):
        max_min_method = {}
        for category in dpm['cv'].keys():
            max_min_method[category] = {}
            for metric in dpm['cv'][category].keys():
                max_min_method[category][metric] = {'max':-np.inf, 'min':np.inf}
        
        for i, method in enumerate(methods_to_plot):
            for category in dpm[method].keys():
                for metric in dpm[method][category].keys():
                    max_min_method[category][metric]['max'] = max(np.max(dpm[method][category][metric]), 
                                                                  max_min_method[category][metric]['max'])
                    max_min_method[category][metric]['min'] = min(np.min(dpm[method][category][metric]),
                                                                   max_min_method[category][metric]['min'])   
        return max_min_method

    max_min_method = find_max_min(driving_performance)
    all_safety = []
    all_comfort = []
    all_efficiency = []
    all_ade = []
    method_performance = np.zeros(len(methods_to_plot))
    for i, method in enumerate(methods_to_plot):
        method_performance[i], ade, safety, comfort, efficiency = \
                plot_method(method, prediction_performance, driving_performance, max_min_method, i, axes, prediction_metric)
        all_ade.extend(ade)
        all_safety.extend(safety)
        all_comfort.extend(comfort)
        all_efficiency.extend(efficiency)
    
    all_ade = np.array(all_ade).reshape(-1,1)
    all_safety = np.array(all_safety).reshape(-1,1)
    all_comfort = np.array(all_comfort).reshape(-1,1)
    all_efficiency = np.array(all_efficiency).reshape(-1,1)

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(all_comfort, all_ade)
    # ade_pred = model.predict(all_comfort)
    # axes[0].plot(ade_pred, all_comfort, color='blue', label='Reverse linear regression')

    # model.fit(all_safety, all_ade)
    # ade_pred = model.predict(all_safety)
    # axes[1].plot(ade_pred, all_safety, color='blue', label='Reverse linear regression')

    # model.fit(all_efficiency, all_ade)
    # ade_pred = model.predict(all_efficiency)
    # axes[2].plot(ade_pred, all_efficiency, color='blue', label='Reverse linear regression')


    # Add a constant term to the predictor variable x

    all_ade_cons = sm.add_constant(all_ade)
    model = sm.OLS(all_safety, all_ade_cons)
    results = model.fit()
    predicted_y = results.predict(all_ade_cons)
    axes[1].plot(all_ade_cons, predicted_y)
    print(results.summary())

    all_ade_cons = sm.add_constant(all_ade)
    model = sm.OLS(all_comfort, all_ade_cons)
    results = model.fit()
    predicted_y = results.predict(all_ade_cons)
    axes[0].plot(all_ade_cons, predicted_y)
    print(results.summary())

    all_ade_cons = sm.add_constant(all_ade)
    model = sm.OLS(all_efficiency, all_ade_cons)
    results = model.fit()
    predicted_y = results.predict(all_ade_cons)
    axes[2].plot(all_ade_cons, predicted_y)
    print(results.summary())

    # Update axis labels
    axes[0].set_title('Comfort')
    axes[0].set_xlabel(f'{prediction_metric}')
    axes[0].set_ylabel('weighted comfort')
    # axes[0].set_xlim([0, x_limit])


    axes[1].set_title('Safety')
    axes[1].set_xlabel(f'{prediction_metric}')
    axes[1].set_ylabel('weighted safety')
    # axes[1].set_xlim([0, x_limit])
    
    axes[2].set_title('Efficiency')
    axes[2].set_xlabel(f'{prediction_metric}')
    axes[2].set_ylabel('weighted efficiency')
    # axes[2].set_xlim([0, x_limit])

    axes[3].set_title('Driving Performance')
    axes[3].set_xlabel(f'{prediction_metric}')
    axes[3].set_ylabel('driving performance')
    # axes[3].set_xlim([0, x_limit])

    # Show the plot
    plt.savefig(f"2_{prediction_metric}_vs_drivingPerformance_DESPOT.png")
    plt.clf()

    return method_performance

if __name__ == "__main__":

    args = parse_args()
    methods = ['cv','ca','hivt','lanegcn','lstmdefault','lstmsocial']
    performance_summit = np.array([2.938, 2.989, 1.692, 1.944, 2.410, 2.480])
    performance_DESPOT = np.zeros(len(methods))
    performance_DESPOT_obs20 = np.zeros(len(methods))
    performance_DESPOT_closest = np.zeros(len(methods))
    
    if args.mode == 'train':
        directories_map_DESPOT = {
            # 'original':'/home/phong/driving_data/official/despot_planner/original_gamma',
            'cv': '/home/phong/driving_data/official/despot_planner/same_computation/cv2Hz',
            'ca': '/home/phong/driving_data/official/despot_planner/same_computation/ca2Hz',
            'hivt': '/home/phong/driving_data/official/despot_planner/same_computation/hivt02Hz',
            'lanegcn': '/home/phong/driving_data/official/despot_planner/same_computation/lanegcn02Hz',
            'lstmdefault': '/home/phong/driving_data/official/despot_planner/same_computation/lstmdefault05Hz',
            'lstmsocial': '/home/phong/driving_data/official/despot_planner/same_computation/lstmsocial03Hz/',
            #'knnsocial': '/home/phong/driving_data/official/despot_planner/smu_server/knnsocial1Hz/',
            #'knndefault': '/home/phong/driving_data/official/despot_planner/smu_server/knndefault1Hz_2times/',
        }

        prediction_performance_DESPOT = {}
        driving_performance_DESPOT = {}

        for key in directories_map_DESPOT.keys():
            print(f"Processing {key}")
            prediction_performance, driving_performance = get_dynamic_ade(directories_map_DESPOT[key])
            prediction_performance_DESPOT[key] = prediction_performance
            driving_performance_DESPOT[key] = driving_performance

        # Save the dictionary to a file
        # save_dict_to_file(prediction_performance_RVO, 'prediction_performance_RVO.pickle')
        # save_dict_to_file(driving_performance_RVO, 'driving_performance_RVO.pickle')
        save_dict_to_file(prediction_performance_DESPOT, f'prediction_performance_DESPOT_{pred_len}.pickle')
        save_dict_to_file(driving_performance_DESPOT, f'driving_performance_DESPOT_{pred_len}.pickle')
    else:
        # Load the dictionary from the file
        # prediction_performance_RVO = load_dict_from_file('prediction_performance_RVO.pickle') 
        # driving_performance_RVO = load_dict_from_file('driving_performance_RVO.pickle')
        prediction_performance_DESPOT = load_dict_from_file(f'prediction_performance_DESPOT_{pred_len}.pickle')
        driving_performance_DESPOT = load_dict_from_file(f'driving_performance_DESPOT_{pred_len}.pickle')
        
    for key in prediction_performance_DESPOT.keys():
        idx = methods.index(key)
        # performance_RVO[idx] = np.nanmean(prediction_performance_RVO[key]['ade'])
        # performance_RVO_obs20[idx] = np.nanmean(prediction_performance_RVO[key]['ade_obs20'])
        # performance_RVO_obs20_closest[idx] = np.nanmean(prediction_performance_RVO[key]['ade_obs20_closest'])
        # print(f"RVO Mean ADE for {key} is {np.nanmean(prediction_performance_RVO[key]['ade']):.2f} and FDE is {np.nanmean(prediction_performance_RVO[key]['fde']):.2f}")
        performance_DESPOT[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade'])
        performance_DESPOT_obs20[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade_obs20'])
        performance_DESPOT_closest[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade_closest'])
        print(f"DESPOT Mean ADE for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade']):.2f} and "
              f"Mean ADE obs20 for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade_obs20']):.2f} and "
              f"Mean ADE closest for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade_closest']):.2f}")


    ## plot static ade vs dynamic ade ##
    offsets=0.2
    position = np.arange(len(methods))
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 9)
    ax.bar(position-1.5*offsets, performance_summit, width=offsets, color='red', align='center', label='Summit_dataset')
    ax.bar(position-0.5*offsets, performance_DESPOT_closest, width=offsets, color='orange', align='center', label='DESPOT_planner_closest')
    ax.bar(position+0.5*offsets, performance_DESPOT_obs20, width=offsets, color='blue', align='center', label='DESPOT_planner_obs20')
    ax.bar(position+1.5*offsets, performance_DESPOT, width=offsets, color='green', align='center', label='DESPOT_planner')
    ax.set_xticks(position)
    ax.set_xticklabels(methods)
    ax.set_title('Comparison of Static_ADE and Dynamic_ADE')
    ax.set_ylabel('ADE')
    ax.legend()
    plt.show()
    plt.savefig("3_Static_ADE_vs_Dynamic_ADE_DESPOT.png")
    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 7)
    # ax.bar(position-offsets, performance_summit, width=offsets, color='red', align='center', label='Summit_dataset')
    # ax.bar(position, performance_RVO_obs20_closest, width=offsets, color='blue', align='center', label='RVO_planner')
    # ax.bar(position+offsets, performance_DESPOT_obs20_closest, width=offsets, color='green', align='center', label='DESPOT_planner')
    # ax.set_xticks(position)
    # ax.set_xticklabels(methods)
    # ax.set_title('Comparison of Static (red) and Dynamic (blue/Green) ADE with 20 obs and closest agent')
    # ax.set_ylabel('ADE')
    # ax.legend()
    # plt.show()
    # plt.savefig("Static_vs_Dynamic_ADE_obs20_closest.png")
    # print("Reason 1: close-loop effect ~ Performance better because agent avoid others")

    # ## Reason 1: close-loop effect ~ Performance better because agent avoid others ##
    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 7)
    # ax.bar(position-offsets, performance_summit, width=offsets, color='red', align='center', label='Summit_dataset')
    # ax.bar(position, performance_DESPOT_obs20, width=offsets, color='blue', align='center', label='DESPOT_planner_obs20')
    # ax.set_xticks(position)
    # ax.set_xticklabels(methods)
    # ax.set_title('Comparison of Static_ADE and Dynamic_ADE_obs20')
    # ax.set_ylabel('ADE')
    # ax.legend()
    # plt.show()
    # plt.savefig("1_Static_vs_Dynamic_ADE_obs20_DESPOT.png")
    # print("Reason 1: close-loop effect ~ Performance better because agent avoid others")
    # print(f"Difference for ['cv','ca','knndefault','knnsocial','hivt','lanegcn','lstmdefault','lstmsocial'] is {performance_DESPOT_obs20-performance_summit} m")
    
    # ## Reason 2: lack observation ~ Performance worse, no need to explain more ##
    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 7)
    # ax.bar(position-offsets, performance_DESPOT_obs20, width=offsets, color='blue', align='center', label='DESPOT_planner_obs20')
    # ax.bar(position, performance_DESPOT, width=offsets, color='#929591', align='center', label='DESPOT_planner')
    # ax.set_xticks(position)
    # ax.set_xticklabels(methods)
    # ax.set_title('Comparison of Dynamic_ADE_obs20 and Dynamic_ADE')
    # ax.set_ylabel('ADE')
    # ax.legend()
    # plt.show()
    # plt.savefig("1_Dynamic_ADE_vs_Dynamic_ADE_obs20_DESPOT.png")
    # print("Reason 2: lack observation ~ Performance worse, no need to explain more")
    # print(f"Difference for ['cv','ca','knndefault','knnsocial','hivt','lanegcn','lstmdefault','lstmsocial'] is {performance_DESPOT - performance_DESPOT_obs20} m")

    # ## Comprehensive Comparison ##
    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 7)
    # ax.bar(position-offsets, performance_summit, width=offsets, color='red', align='center', label='Summit_dataset')
    # ax.bar(position, performance_DESPOT, width=offsets, color='#929591', align='center', label='DESPOT_planner')
    # # ax.bar(position+offsets, performance_DESPOT, width=offsets, color='green', align='center', label='DESPOT_planner')
    # ax.set_xticks(position)
    # ax.set_xticklabels(methods)
    # ax.set_title('Comparison of Static_ADE and Dynamic_ADE')
    # ax.set_ylabel('ADE')
    # ax.legend()
    # plt.show()
    # plt.savefig("1_Static_vs_Dynamic_ADE_DESPOT.png")
    # print("Comprehensive Comparison Done")
    # print(f"Difference for ['cv','ca','knndefault','knnsocial','hivt','lanegcn','lstmdefault','lstmsocial'] is {performance_DESPOT - performance_summit} m")
    method_performance = scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = prediction_performance_DESPOT.keys(), \
                                    prediction_metric = 'ade', x_limit = 1)
    ## Static ADE vs Driving Performance##
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.scatter(x=performance_summit, y=method_performance, c='red', marker='o')
    # Add labels for each point
    for i in range(len(method_performance)):
        ax.text(performance_summit[i], method_performance[i], methods[i])
    ax.set_xlabel("Static ADE")
    ax.set_title('Relation between Static ADE and Driving Performance')
    ax.set_ylabel('Driving Performance')
    plt.show()
    plt.savefig("3_Static_ADE_vs_Driving_Performance_DESPOT.png")
    print("Plot the relation between Static ADE and Driving Performance")

    ## Dynamic ADE vs Driving Performance##
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    performance_dynamic = [np.nanmean(prediction_performance_DESPOT[key]['ade']) for key in methods]
    ax.scatter(x=performance_dynamic, y=method_performance, c='red', marker='o')
    # Add labels for each point
    for i in range(len(method_performance)):
        ax.text(performance_dynamic[i], method_performance[i], methods[i])
    ax.set_xlabel("Dynamic ADE")
    ax.set_title('Relation between Dynamic ADE and Driving Performance')
    ax.set_ylabel('Driving Performance')
    plt.show()
    plt.savefig("3_Dynamic_ADE_vs_Driving_Performance_DESPOT.png")
    print("Plot the relation between Dynamic ADE and Driving Performance")
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'ade_obs20', x_limit = 1)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'ade_closest', x_limit = 1)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'std', x_limit = 1)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'std_closest', x_limit = 1)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'temp_consistency', x_limit = 1)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'temp_consistency_closest', x_limit = 1)



