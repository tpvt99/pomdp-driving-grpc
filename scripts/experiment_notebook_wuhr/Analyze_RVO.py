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

from common_performance_stats import get_dynamic_ade, get_dynamic_ade_eachmap

# Weights for each metric within a category
safety_weights = {
    'collision_rate': 1.0,
    'near_miss_rate': 0.0,
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
    
    # max_values = [max(prediction_performance[name][prediction_metric]) for name in prediction_performance.keys()]
    # min_values = [min(prediction_performance[name][prediction_metric]) for name in prediction_performance.keys()]
    # prediction_data = normalize(np.array(prediction_performance[method][prediction_metric]), max(max_values), min(min_values))
    
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

    all_ade_1, safety_data_1 = remove_outliers_iqr(all_ade, all_safety, multiplier=1.5)
    all_ade_2, comfort_data_2 = remove_outliers_iqr(all_ade, all_comfort, multiplier=1.5)
    all_ade_3, efficiency_data_3 = remove_outliers_iqr(all_ade, all_efficiency, multiplier=1.5)

    
    interval1 = (np.max(all_ade_1)-np.min(all_ade_1))/30
    bins1 = np.arange(np.min(all_ade_1),np.max(all_ade_1), interval1)
    bin_indices1 = np.digitize(all_ade_1, bins1)

    interval2 = (np.max(all_ade_2)-np.min(all_ade_2))/30
    bins2 = np.arange(np.min(all_ade_2),np.max(all_ade_2), interval2)
    bin_indices2 = np.digitize(all_ade_2, bins2)

    interval3 = (np.max(all_ade_3)-np.min(all_ade_3))/30
    bins3 = np.arange(np.min(all_ade_3),np.max(all_ade_3), interval3)
    bin_indices3 = np.digitize(all_ade_3, bins3)

    all_safety_means, all_comfort_means, all_efficiency_means = [], [], []
    for i in range(1, len(bins1)):
        all_safety_means.append(np.mean(safety_data_1[bin_indices1 == i]))
    for i in range(1, len(bins2)):
        all_comfort_means.append(np.mean(comfort_data_2[bin_indices2 == i]))
    for i in range(1, len(bins3)):
        all_efficiency_means.append(np.mean(efficiency_data_3[bin_indices3 == i]))
    
    all_ade_means1 = bins1[:-1] + interval1/2
    all_ade_means2 = bins2[:-1] + interval2/2
    all_ade_means3 = bins3[:-1] + interval3/2

    axes[0].scatter(x=all_ade_means2, y=all_comfort_means, c='red', marker='o')
    axes[1].scatter(x=all_ade_means1, y=all_safety_means, c='red', marker='o')
    axes[2].scatter(x=all_ade_means3, y=all_efficiency_means, c='red', marker='o')
    
    # import pdb;pdb.set_trace()


    
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

    # all_comfort_cons = sm.add_constant(comfort_data_2)
    # model = sm.OLS(all_ade_2, all_comfort_cons)
    # results = model.fit(loss='mse')
    # predicted_ade = results.predict(all_comfort_cons)
    # axes[0].plot(predicted_ade, all_comfort_cons[:,1])
    # print(results.summary())

    # all_safety_cons = sm.add_constant(safety_data_1)
    # model = sm.OLS(all_ade_1, all_safety_cons)
    # results = model.fit(loss='mse')
    # predicted_ade = results.predict(all_safety_cons)
    # axes[1].plot(predicted_ade, all_safety_cons[:,1])
    # print(results.summary())

    # all_efficiency_cons = sm.add_constant(efficiency_data_3)
    # model = sm.OLS(all_ade_3, all_efficiency_cons)
    # results = model.fit(loss='mse')
    # predicted_ade = results.predict(all_efficiency_cons)
    # axes[2].plot(predicted_ade, all_efficiency_cons[:,1])
    # print(results.summary())

    # Update axis labels

    # axes[0].scatter(x=all_ade_2, y=comfort_data_2, c='red', marker='o')
    # axes[1].scatter(x=all_ade_1, y=safety_data_1, c='red', marker='o')
    # axes[2].scatter(x=all_ade_3, y=efficiency_data_3, c='red', marker='o')

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

    # Show the plot
    plt.savefig(f"0_{prediction_metric}_vs_drivingPerformance_RVO.png")
    plt.clf()

def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = {}
    for key in ['safety','comfort','efficiency','performance']:
        method_performance[key] = []

    for i, method in enumerate(methods_to_plot):
        _, safety, comfort, efficiency, performance = get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric)
        method_performance['safety'].append(np.mean(safety))
        method_performance['comfort'].append(np.mean(comfort))
        method_performance['efficiency'].append(np.mean(efficiency))
        method_performance['performance'].append(np.mean(performance))

    return method_performance


if __name__ == "__main__":


    args = parse_args()

    methods = ['cv','ca','knndefault','knnsocial','hivt','lanegcn','lstmdefault','lstmsocial']
    performance_summit = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_RVO = np.zeros(len(methods))
    performance_RVO_obs20 = np.zeros(len(methods))
    performance_RVO_closest = np.zeros(len(methods))

    if args.mode == 'train':

        # directories_map_RVO = {
        #     'cv': '/home/phong/driving_data/official/gamma_planner_path1_vel3/cv/result/gamma_drive_mode/shi_men_er_lu/',
        #     'ca': '/home/phong/driving_data/official/gamma_planner_path1_vel3/ca/result/gamma_drive_mode/shi_men_er_lu/',
        #     'hivt': '/home/phong/driving_data/official/gamma_planner_path1_vel3/hivt/result/gamma_drive_mode/shi_men_er_lu/',
        #     'lanegcn': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lanegcn/result/gamma_drive_mode/shi_men_er_lu/',
        #     'lstmdefault': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmdefault/result/gamma_drive_mode/shi_men_er_lu/',
        #     'lstmsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmsocial5Hz/result/gamma_drive_mode/shi_men_er_lu/',
        #     'knnsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/knnsocial/result/gamma_drive_mode/shi_men_er_lu/',
        #     'knndefault': '/home/phong/driving_data/official/gamma_planner_path1_vel3/knndefault/result/gamma_drive_mode/shi_men_er_lu/'
        # }

        directories_map_RVO = {
            'cv': '/home/phong/driving_data/official/gamma_planner_path1_vel3/cv',
            'ca': '/home/phong/driving_data/official/gamma_planner_path1_vel3/ca',
            'hivt': '/home/phong/driving_data/official/gamma_planner_path1_vel3/hivt',
            'lanegcn': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lanegcn',
            'lstmdefault': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmdefault',
            'lstmsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmsocial5Hz',
            'knnsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/knnsocial',
            'knndefault': '/home/phong/driving_data/official/gamma_planner_path1_vel3/knndefault'
        }

        prediction_performance_RVO = {}
        driving_performance_RVO = {}

        for key in directories_map_RVO.keys():
            print(f"Processing {key}")
            prediction_performance, driving_performance = get_dynamic_ade(directories_map_RVO[key], pred_len)
            # prediction_performance, driving_performance = get_dynamic_ade_eachmap(directories_map_RVO[key], pred_len)
            prediction_performance_RVO[key] = prediction_performance
            driving_performance_RVO[key] = driving_performance

        # Save the dictionary to a file
        save_dict_to_file(prediction_performance_RVO, f'prediction_performance_RVO_{pred_len}.pickle')
        save_dict_to_file(driving_performance_RVO, f'driving_performance_RVO_{pred_len}.pickle')
    else:
        # Load the dictionary from the file
        prediction_performance_RVO = load_dict_from_file(f'prediction_performance_RVO_{pred_len}.pickle') 
        driving_performance_RVO = load_dict_from_file(f'driving_performance_RVO_{pred_len}.pickle')
    
    for key in prediction_performance_RVO.keys():
        idx = methods.index(key)
        performance_RVO[idx] = np.nanmean(prediction_performance_RVO[key]['ade'])
        performance_RVO_obs20[idx] = np.nanmean(prediction_performance_RVO[key]['ade_obs20'])
        performance_RVO_closest[idx] = np.nanmean(prediction_performance_RVO[key]['ade_closest'])
        print(f"RVO Mean ADE for {key} is {np.nanmean(prediction_performance_RVO[key]['ade']):.2f} "
              f"Mean ADE obs20 for {key} is {np.nanmean(prediction_performance_RVO[key]['ade_obs20']):.2f} " 
              f"Mean ADE obs20 closest for {key} is {np.nanmean(prediction_performance_RVO[key]['ade_closest']):.2f}")
    
    
    # for metric in prediction_performance_RVO['cv'].keys():
    #     method_performance = get_method_performance(prediction_performance_RVO, driving_performance_RVO, methods, metric)

    #     fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    #     performance_dynamic = [np.nanmean(prediction_performance_RVO[method][metric]) for method in methods]
    #     for i, eval_metric in enumerate(method_performance.keys()):
    #         ax[i].scatter(x=performance_dynamic, y=method_performance[eval_metric], c='red', marker='o')
    #     # Add labels for each point
    #         for j in range(len(methods)):
    #             ax[i].text(performance_dynamic[j], method_performance[eval_metric][j], methods[j])
    #         ax[i].set_xlabel(metric)
    #         ax[i].set_ylabel(eval_metric)
    #         ax[i].set_title(f'Relation between {metric} and {eval_metric}')

    #     plt.show()
    #     plt.savefig(f"0_{metric}_RVO.png")

    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'ade', x_limit = pred_len/5)

    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'ade_obs20', x_limit = pred_len/5)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'ade_closest', x_limit = pred_len/5)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'std', x_limit = pred_len/7.5)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'std_closest', x_limit = pred_len/7.5)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'temp_consistency', x_limit = pred_len/150)
    
    scatter_plot_multi_pred_3_metric(prediction_performance_RVO, driving_performance_RVO, methods_to_plot = methods, \
                                    prediction_metric = 'temp_consistency_closest', x_limit = pred_len/150)






