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
import pandas as pd
import statsmodels.api as sm
import numpy as np

from common_performance_stats import get_dynamic_ade

# Weights for each metric within a category
safety_weights = {
    'collision_rate': 0.0,
    'near_miss_rate': 1.0,
    'near_distance_rate': 0.0
}
safety_directions = {
    'collision_rate': 'lower',
    'near_miss_rate': 'lower',
    'near_distance_rate': 'lower'
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

pred_len = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")

    parser.add_argument('--mode', help='Generate file or only plot the relation', required=True)

    args = parser.parse_args()

    return args

def save_dict_to_file(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dictionary, f)

# Load the dictionary from a file
def load_dict_from_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def remove_outliers_iqr(X, y1, multiplier=1):
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
    
    # Calculate the IQR for X and y
    Q1_X = np.percentile(X, 25)
    Q3_X = np.percentile(X, 75)
    IQR_X = Q3_X - Q1_X

    Q1_y1 = np.percentile(y1, 25)
    Q3_y1 = np.percentile(y1, 75)
    IQR_y1 = Q3_y1 - Q1_y1


    # Define the upper and lower bounds for outlier detection
    upper_bound_X = Q3_X + multiplier * IQR_X
    lower_bound_X = Q1_X - multiplier * IQR_X

    upper_bound_y1 = Q3_y1 + multiplier * IQR_y1
    lower_bound_y1 = Q1_y1 - multiplier * IQR_y1

    # Filter the data to keep only the values within the bounds
    X_clean = X[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) & (y1!=0)]
    y1_clean = y1[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) & (y1!=0)]

    return X_clean, y1_clean

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

def find_max_min(dpm, methods_to_plot):
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

def get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric):

    # Compute the weighted averages for each category
    max_min_safety = [tuple(max_min_method['safety'][key].values()) for key in max_min_method['safety'].keys()]
    safety_data = weighted_average(
        [driving_performance[method]['safety']['collision_rate'],
        driving_performance[method]['safety']['near_miss_rate'],
        driving_performance[method]['safety']['near_distance_rate']],
        list(safety_weights.values()), list(safety_directions.values()),
        max_min_safety)

    max_min_comfort = [tuple(max_min_method['comfort'][key].values()) for key in max_min_method['comfort'].keys()]
    comfort_data = weighted_average(
        [driving_performance[method]['comfort']['jerk'],
        driving_performance[method]['comfort']['lateral_acceleration'],
        driving_performance[method]['comfort']['acceleration']],
        list(comfort_weights.values()), list(comfort_directions.values()), max_min_comfort)

    max_min_efficiency = [tuple(max_min_method['efficiency'][key].values()) for key in max_min_method['efficiency'].keys()]
    efficiency_data = weighted_average(
        [driving_performance[method]['efficiency']['avg_speed'],
        driving_performance[method]['efficiency']['tracking_error'],
        driving_performance[method]['efficiency']['efficiency_time'],
        driving_performance[method]['efficiency']['distance_traveled']],
        list(efficiency_weights.values()), list(efficiency_directions.values()), max_min_efficiency)
    
    prediction_data = np.array(prediction_performance[method][prediction_metric])

    driving_performance_data = (efficiency_data+safety_data+comfort_data)/3

    return prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data
    

def scatter_plot_multi_pred_3_metric(prediction_performance, driving_performance, methods_to_plot, prediction_metric, x_limit):  

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    max_min_method = find_max_min(driving_performance, methods_to_plot)

    all_ade = []
    all_safety = []
    all_comfort = []
    all_efficiency = []
    for i, method in enumerate(methods_to_plot):
        ade, safety, comfort, efficiency,_ = get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric)
        all_ade.extend(ade)
        all_safety.extend(safety)
        all_comfort.extend(comfort)
        all_efficiency.extend(efficiency)
    all_ade = np.array(all_ade).reshape(-1,1)
    all_safety = np.array(all_safety).reshape(-1,1)
    all_comfort = np.array(all_comfort).reshape(-1,1)
    all_efficiency = np.array(all_efficiency).reshape(-1,1)

    all_ade_1, safety_data_1 = remove_outliers_iqr(all_ade, all_safety, multiplier=3)
    all_ade_2, comfort_data_2 = remove_outliers_iqr(all_ade, all_comfort, multiplier=3)
    all_ade_3, efficiency_data_3 = remove_outliers_iqr(all_ade, all_efficiency, multiplier=3)
    
    # all_ade_cons = sm.add_constant(all_ade_1)
    # model = sm.OLS(safety_data_1, all_ade_cons)
    # results = model.fit(loss='mse')
    # predicted_y = results.predict(all_ade_cons)
    # axes[1].plot(all_ade_cons[:,1], predicted_y)
    # print(results.summary())

    # all_ade_cons = sm.add_constant(all_ade_2)
    # model = sm.OLS(comfort_data_2, all_ade_cons)
    # results = model.fit(loss='mse')
    # predicted_y = results.predict(all_ade_cons)
    # axes[0].plot(all_ade_cons[:,1], predicted_y)
    # print(results.summary())

    # all_ade_cons = sm.add_constant(all_ade_3)
    # model = sm.OLS(efficiency_data_3, all_ade_cons)
    # results = model.fit(loss='mse')
    # predicted_y = results.predict(all_ade_cons)
    # axes[2].plot(all_ade_cons[:,1], predicted_y)
    # print(results.summary())

    all_comfort_cons = sm.add_constant(comfort_data_2)
    model = sm.OLS(all_ade_2, all_comfort_cons)
    results = model.fit(loss='mse')
    predicted_ade = results.predict(all_comfort_cons)
    axes[0].plot(predicted_ade, all_comfort_cons[:,1])
    print(results.summary())

    all_safety_cons = sm.add_constant(safety_data_1)
    model = sm.OLS(all_ade_1, all_safety_cons)
    results = model.fit(loss='mse')
    predicted_ade = results.predict(all_safety_cons)
    axes[1].plot(predicted_ade, all_safety_cons[:,1])
    print(results.summary())

    all_efficiency_cons = sm.add_constant(efficiency_data_3)
    model = sm.OLS(all_ade_3, all_efficiency_cons)
    results = model.fit(loss='mse')
    predicted_ade = results.predict(all_efficiency_cons)
    axes[2].plot(predicted_ade, all_efficiency_cons[:,1])
    print(results.summary())

    # Update axis labels

    axes[0].scatter(x=all_ade_2, y=comfort_data_2, c='red', marker='o')
    axes[1].scatter(x=all_ade_1, y=safety_data_1, c='red', marker='o')
    axes[2].scatter(x=all_ade_3, y=efficiency_data_3, c='red', marker='o')

    axes[0].set_title('Comfort')
    axes[0].set_xlabel(f'{prediction_metric}')
    axes[0].set_ylabel('weighted comfort')
    axes[0].set_xlim([0, x_limit])


    axes[1].set_title('Safety')
    axes[1].set_xlabel(f'{prediction_metric}')
    axes[1].set_ylabel('weighted safety')
    axes[1].set_xlim([0, x_limit])
    
    axes[2].set_title('Efficiency')
    axes[2].set_xlabel(f'{prediction_metric}')
    axes[2].set_ylabel('weighted efficiency')
    axes[2].set_xlim([0, x_limit])

    # Show the plot
    plt.savefig(f"2_{prediction_metric}_vs_drivingPerformance_DESPOT.png")
    plt.clf()

    return method_performance

def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = np.zeros(len(methods_to_plot))
    for i, method in enumerate(methods_to_plot):
        _, _, _, _, performance = get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric)
        method_performance[i] = np.mean(performance)

    return method_performance

if __name__ == "__main__":

    args = parse_args()
    methods = ['cv','ca','knndefault','knnsocial','hivt','lanegcn','lstmdefault','lstmsocial']
    performance_summit = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_DESPOT = np.zeros(len(methods))
    performance_DESPOT_obs20 = np.zeros(len(methods))
    performance_DESPOT_closest = np.zeros(len(methods))
    
    if args.mode == 'train':
        directories_map_DESPOT = {
            'cv': '/home/phong/driving_data/official/despot_planner/same_computation/cv2Hz',
            'ca': '/home/phong/driving_data/official/despot_planner/same_computation/ca2Hz',
            'hivt': '/home/phong/driving_data/official/despot_planner/same_computation/hivt02Hz',
            'lanegcn': '/home/phong/driving_data/official/despot_planner/same_computation/lanegcn02Hz',
            'lstmdefault': '/home/phong/driving_data/official/despot_planner/same_computation/lstmdefault05Hz',
            'lstmsocial': '/home/phong/driving_data/official/despot_planner/same_computation/lstmsocial03Hz/',
            'knnsocial': '/home/phong/driving_data/official/despot_planner/same_computation/knnsocial01Hz/',
            'knndefault': '/home/phong/driving_data/official/despot_planner/same_computation/knndefault01Hz/',
        }

        prediction_performance_DESPOT = {}
        driving_performance_DESPOT = {}

        for key in directories_map_DESPOT.keys():
            print(f"Processing {key}")
            prediction_performance, driving_performance = get_dynamic_ade(directories_map_DESPOT[key], pred_len)
            prediction_performance_DESPOT[key] = prediction_performance
            driving_performance_DESPOT[key] = driving_performance

        save_dict_to_file(prediction_performance_DESPOT, f'prediction_performance_DESPOT_{pred_len}.pickle')
        save_dict_to_file(driving_performance_DESPOT, f'driving_performance_DESPOT_{pred_len}.pickle')
    else:
        prediction_performance_DESPOT = load_dict_from_file(f'prediction_performance_DESPOT_{pred_len}.pickle')
        driving_performance_DESPOT = load_dict_from_file(f'driving_performance_DESPOT_{pred_len}.pickle')
        
    for key in prediction_performance_DESPOT.keys():
        idx = methods.index(key)
        performance_DESPOT[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade'])
        performance_DESPOT_obs20[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade_obs20'])
        performance_DESPOT_closest[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade_closest'])
        print(f"DESPOT Mean ADE for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade']):.2f} and "
              f"Mean ADE obs20 for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade_obs20']):.2f} and "
              f"Mean ADE closest for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade_closest']):.2f}")

    method_performance = get_method_performance(prediction_performance_DESPOT, driving_performance_DESPOT, methods, 'ade')

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

    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = prediction_performance_DESPOT.keys(), \
                                    prediction_metric = 'ade', x_limit = pred_len/4)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'ade_obs20', x_limit = pred_len/4)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'ade_closest', x_limit = pred_len/4)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'std', x_limit = pred_len/4)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'std_closest', x_limit = pred_len/4)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'temp_consistency', x_limit = pred_len/150)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_DESPOT, driving_performance_DESPOT, methods_to_plot = methods, \
                                    prediction_metric = 'temp_consistency_closest', x_limit = pred_len/150)



