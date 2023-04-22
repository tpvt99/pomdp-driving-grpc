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
import textwrap

from common_performance_stats import get_dynamic_ade

# Weights for each metric within a category
safety_weights = {
    'collision_rate': 0.2,
    'near_miss_rate': 0.5,
    'near_distance_rate': 0.0,
    'mean_min_ttc': 0.0,
    'mean_min_ttr': 0.3
}
safety_directions = {
    'collision_rate': 'lower',
    'near_miss_rate': 'lower',
    'near_distance_rate': 'lower',
    'mean_min_ttc': 'higher',
    'mean_min_ttr': 'higher'
}
comfort_weights = {
    'jerk': 0.1,
    'lateral_acceleration': 0.8,
    'acceleration': 0.1
}
comfort_directions = {
    'jerk': 'lower',
    'lateral_acceleration': 'lower',
    'acceleration': 'lower'
}
efficiency_weights = {
    'avg_speed': 0.8,
    'tracking_error': 0.2,
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

    # Mode 'data' is to generate data from .txt files
    # Mode 'performance' is to read data pickle file to generate prediction and performance, tree metrics
    # Mode 'plot' is to read from metric pickle file and plot
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

def remove_nans(list_of_lists):
    
    # Initialize an empty list to hold the indices of elements to remove
    indices_to_remove = []

    # Iterate through each inner list and find indices containing nan values
    for inner_list in list_of_lists:
        for i, element in enumerate(inner_list):
            if math.isnan(element):
                indices_to_remove.append(i)

    # Remove duplicate indices from the list and reverse sort it
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)

    # Iterate through the list of indices to remove and remove elements from the inner lists
    for i in indices_to_remove:
        for inner_list in list_of_lists:
            del inner_list[i]

    return list_of_lists

def normalize(arr, max_val = 1.0, min_val = 0.0):
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    
    return (arr - np.array(min_val)) / (np.array(max_val) - np.array(min_val)+1e-6)

def weighted_average(data, weights, directions, max_min_list):
    norm_data = [normalize(arr, max_min[0], max_min[1]) if direction == 'higher'
                    else 1 - normalize(arr, max_min[0], max_min[1]) for arr, direction, max_min in zip(data, directions, max_min_list)]
    return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)

def weighted_average_no_normalize(data, weights, directions, max_min_list):
    norm_data = [normalize(arr, 1, 0) for arr, direction, max_min in zip(data, directions, max_min_list)]
    return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)

def find_max_min(dpm, methods_to_plot):
    max_min_method = {}
    a_method_in_dpm = list(dpm.keys())[0]
    for category in dpm[a_method_in_dpm].keys():
        max_min_method[category] = {}
        for metric in dpm[a_method_in_dpm][category].keys():
            max_min_method[category][metric] = {'max':-np.inf, 'min':np.inf}
    
    for i, method in enumerate(methods_to_plot):
        for category in dpm[method].keys():
            for metric in dpm[method][category].keys():
                max_min_method[category][metric]['max'] = max(np.nanmax(dpm[method][category][metric]), 
                                                                max_min_method[category][metric]['max'])
                max_min_method[category][metric]['min'] = min(np.nanmin(dpm[method][category][metric]),
                                                                max_min_method[category][metric]['min'])   
    return max_min_method

def get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric):

    # Compute the weighted averages for each category
    max_min_safety = []
    w = []
    d = []
    assert len(list(max_min_method['safety'].keys())) == 5, "Must be 5"
    assert list(max_min_method['safety'].keys()) == ["collision_rate", "near_miss_rate", 
                                                     "near_distance_rate", "mean_min_ttc" , "mean_min_ttr"]
    for k in ["collision_rate", "near_miss_rate", "near_distance_rate", "mean_min_ttc" , "mean_min_ttr"]:
        max_min_safety.append(tuple(max_min_method['safety'][k].values()))
        w.append(safety_weights[k])
        d.append(safety_directions[k])

    safety_data = weighted_average(
        [driving_performance[method]['safety']['collision_rate'],
        driving_performance[method]['safety']['near_miss_rate'],
        driving_performance[method]['safety']['near_distance_rate'],
        driving_performance[method]['safety']['mean_min_ttc'],
        driving_performance[method]['safety']['mean_min_ttr']],
        w, d, max_min_safety)


    max_min_comfort = []
    w = []
    d = []
    assert len(list(max_min_method['comfort'].keys())) == 3, "Must be 3"
    assert list(max_min_method['comfort'].keys()) == ["jerk", "lateral_acceleration", "acceleration"]
    for k in ["jerk", "lateral_acceleration", "acceleration"]:
        max_min_comfort.append(tuple(max_min_method['comfort'][k].values()))
        w.append(comfort_weights[k])
        d.append(comfort_directions[k])
    
    comfort_data = weighted_average(
        [driving_performance[method]['comfort']['jerk'],
        driving_performance[method]['comfort']['lateral_acceleration'],
        driving_performance[method]['comfort']['acceleration']],
        w, d, max_min_comfort)

    max_min_efficiency = []
    w = []
    d = []
    assert len(list(max_min_method['efficiency'].keys())) == 4, "Must be 4"
    assert list(max_min_method['efficiency'].keys()) == ["avg_speed", "tracking_error", "efficiency_time", "distance_traveled"]
    for k in ["avg_speed", "tracking_error", "efficiency_time", "distance_traveled"]:
        max_min_efficiency.append(tuple(max_min_method['efficiency'][k].values()))
        w.append(efficiency_weights[k])
        d.append(efficiency_directions[k])
    
    efficiency_data = weighted_average(
        [driving_performance[method]['efficiency']['avg_speed'],
        driving_performance[method]['efficiency']['tracking_error'],
        driving_performance[method]['efficiency']['efficiency_time'],
        driving_performance[method]['efficiency']['distance_traveled']],
        w, d, max_min_efficiency)
    
    prediction_data = np.array(prediction_performance[method][prediction_metric])

    driving_performance_data = (efficiency_data+safety_data+comfort_data)/3

    return prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data



def plot_driving_performance_sameHz(prediction_performance_map, driving_performance_map, 
                                    algorithm_performance_DESPOT,
                                    methods_to_plot, driving_metric='driving'):
    # methods to plot is like [cv30Hz, cv1Hz, cv3Hz]
    prediction_metric = 'ade' # Do not care

    max_min_method = find_max_min(driving_performance_map, methods_to_plot)
    
    performance_mean = {}

    for method in methods_to_plot:
        prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data = \
             get_normalized_data(method, prediction_performance_map, driving_performance_map, max_min_method, prediction_metric)
        performance_mean[method] = {'safety': np.nanmean(safety_data),
                                'comfort': np.nanmean(comfort_data),
                                'efficiency': np.nanmean(efficiency_data),
                                'driving': np.nanmean(driving_performance_data)}

    bar_width = 0.2
    opacity = 0.8

    frequencies = ['30Hz', '3Hz', '1Hz']
    methods = ['cv', 'ca', 'lstmdefault', 'lstmsocial', 'hivt', 'lanegcn', 'knndefault', 'knnsocial']

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    index = np.arange(len(frequencies))
    markers = ['o', 's', 'D', 'v', '^', '>', '<', 'p']  # List of markers for each method

    # Plot combined frequencies
    fig, ax1 = plt.subplots()
    for i, method in enumerate(methods):
        performance = []

        for j, freq in enumerate(frequencies):
            print(f"Inside plot: {performance_mean[f'{method}{freq}']}")
            assert f"{method}{freq}" in performance_mean.keys()
            if np.isnan(performance_mean[f"{method}{freq}"][driving_metric]):
                performance.append(0.0)
            else:
                performance.append(performance_mean[f"{method}{freq}"][driving_metric])

        ax1.plot(index, performance, label=f'{method}', color=default_colors[i % len(default_colors)], 
                 marker=markers[i], markersize = 5)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Driving Performance')
    ax1.set_title('Driving Performance at Different Planning Frequencies')
    ax1.set_xticks(index)
    ax1.set_xticklabels(frequencies)
    ax1.legend()
    ax1.set_ylim([0.3, 0.9])

    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Display the legend horizontally using ncol parameter
    ax1.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.15))



    # Wrap the long method names to fit within a certain width
    # wrapped_methods = []
    # for method in methods:
    #     if 'knn' in method.lower():
    #         wrapped_methods.append(method.replace('knn', 'knn\n', 1))
    #     elif 'lstm' in method.lower():
    #         wrapped_methods.append(method.replace('lstm', 'lstm\n', 1))
    #     else:
    #         wrapped_methods.append(textwrap.fill(method, 12))


    # plt.xlabel('Motion Prediction Methods')
    # plt.ylabel(f'{driving_metric.capitalize()} Performance')
    # plt.title(f'{driving_metric.capitalize()} Performance Comparison for Different Frequencies')
    # plt.xticks(index + bar_width, wrapped_methods, rotation=0)
    # plt.legend()
    # plt.tight_layout()
    #plt.show()

    plt.savefig(f"1_DESPOT_sameHz_{driving_metric}performance.png")
    plt.clf()


# def plot_driving_performance_sameHz_issue1(prediction_performance_map, driving_performance_map, tree_performance_map,
#                                            methods_to_plot, driving_metric='driving', plot_type='bar'):
#     # methods to plot is like [cv30Hz, cv1Hz, cv3Hz]
#     prediction_metric = 'fde' 

#     max_min_method = find_max_min(driving_performance_map, methods_to_plot)
    
#     performance_mean = {}

#     for method in methods_to_plot:
#         prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data = \
#              get_normalized_data(method, prediction_performance_map, driving_performance_map, max_min_method, prediction_metric)
#         performance_mean[method] = {'safety': np.nanmean(safety_data),
#                                 'comfort': np.nanmean(comfort_data),
#                                 'efficiency': np.nanmean(efficiency_data),
#                                 'driving': np.nanmean(driving_performance_data)}
#     ade_performance = {}
#     # Printing out tree performance with only metric we want a secondary plot
#     for method in methods_to_plot:
#         for metric in prediction_performance_map[method].keys():
#             if metric == prediction_metric:
#                 values =  prediction_performance_map[method][metric]
#                 # if isinstance(values, int):
#                 #     print(f"Method {method} Metric {metric} with Value: {values:.2f}")
#                 # elif isinstance(values, list) and len(values) == 0:
#                 #     print(f'Error at {method} {metric}')
#                 # elif isinstance(values[0], list):
#                 #     flattened_list = [value for sublist in values for value in sublist]
#                 #     print(f"Method {method} Metric {metric} with Average Value: {np.mean(np.array(flattened_list)):.2f}")
#                 # else:
#                 #     print(f"Method {method} Metric {metric} with Average Value: {np.mean(np.array(values)):.2f}")
#                 ade_performance[method] = np.mean(np.array(values))

#     bar_width = 0.35
#     opacity = 0.8

#     frequency = '1Hz'
#     methods = ['cv', 'ca', 'lstmdefault', 'lstmsocial', 'hivt', 'lanegcn', 'knndefault', 'knnsocial']
#     default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    
#     # Wrap the long method names to fit within a certain width
#     wrapped_methods = []
#     for method in methods:
#         if 'knn' in method.lower():
#             wrapped_methods.append(method.replace('knn', 'knn\n', 1))
#         elif 'lstm' in method.lower():
#             wrapped_methods.append(method.replace('lstm', 'lstm\n', 1))
#         else:
#             wrapped_methods.append(textwrap.fill(method, 12))

#     index = np.arange(len(methods))

#     #Plot combined frequencies
#     fig, ax1 = plt.subplots()
#     #fig.subplots_adjust(bottom=0.2, top=0.9)  # Add this line to adjust the top and bottom margins of the plot

#     for i, method in enumerate(methods):
#         performance = []
#         colors = []
        
#         print(f"Inside plot: {performance_mean[f'{method}{frequency}']}")
#         assert f"{method}{frequency}" in performance_mean.keys()
#         if np.isnan(performance_mean[f"{method}{frequency}"][driving_metric]):
#             colors.append('red')
#             performance.append(0.0)
#         else:
#             colors.append(default_colors[i % 3]) # use a different color for each frequency
#             performance.append(performance_mean[f"{method}{frequency}"][driving_metric])
        
#         ax1.bar(index[i], performance, bar_width, alpha=opacity, label=f'{method}')
#         for j, color in enumerate(colors):
#             if color == 'red':
#                 plt.plot(index[i], 0.5, marker='x', markersize=5, color='red')

#         #ax1.set_xlabel('Frequency (Hz)')
#         ax1.set_ylabel('Driving Performance')
#         ax1.yaxis.label.set_color('red')  # Add this line to set the color of the Driving Performance label
#         ax1.set_title(f"Driving Performance at 0.1Hz Planning of 8 Prediction Models.\n "\
#                       "Bars are sorted by Time Per Moped")
#         ax1.set_xticks(index)
#         #ax1.set_xticklabels(methods)
#         ax1.set_xticklabels(wrapped_methods)
#         ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(methods))
#         ax1.set_ylim([0, 1])

#          # Add this block of code after plotting the bar using ax1.bar()
#         # y_value = performance[0]
#         # x_offset = index[i] - bar_width/6
#         # y_offset = y_value + 0.01
#         # ax1.text(x_offset, y_offset, f'{y_value:.2f}', ha='center', va='bottom', fontsize=8)


#     # Adding text annotations to indicate better values
#     ax1.annotate('Higher is better', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10, color='red', bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=0.2'))
#     ax1.annotate('Lower is better', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, color='blue', bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round,pad=0.2'))
#     ax1.set_ylim([0, 1.1])  # Add this line to adjust the inner spacing of the plot by setting y-axis limits


#     # Creating the line plot on the secondary axis
#     ax2 = ax1.twinx()
#     ade_values = [ade_performance[f"{method}{frequency}"] for method in methods]
#     print(ade_values)
#     ax2.plot(np.arange(len(ade_values)), ade_values, 'bo-', 
#              label=f'Dynamic {"ADE" if prediction_metric == "ade" else "FDE"}' if i == 0 else "")
#     ax2.set_ylabel(f'Dynamic {"ADE" if prediction_metric == "ade" else "FDE"}')
#     ax2.yaxis.label.set_color('blue')  # Add this line to set the color of the Depth Tree label
#     ax2.tick_params(axis='y')#, labelcolor='blue')
#     #ax2.legend(loc='upper right')

#     # Set y-lim
#     min_ade_values = min(ade_values)
#     max_ade_values = max(ade_values)
#     range_ade_tree = max_ade_values - min_ade_values
#     ax2.set_ylim(min_ade_values - 0.1 * range_ade_tree, max_ade_values + 0.1 * range_ade_tree)  # Adjust the secondary y-axis limits



#     plt.savefig(f"2_DESPOT_01Hz_{driving_metric}performance_{prediction_metric}.png")
#     plt.clf()

def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = np.zeros(len(methods_to_plot))
    for i, method in enumerate(methods_to_plot):
        _, _, _, _, performance = get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric)
        method_performance[i] = np.mean(performance)

    return method_performance

if __name__ == "__main__":

    args = parse_args()

    directories_map_DESPOT = {
            'cv30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/cv30Hz',
            'cv3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/cv3Hz',
            'cv1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/cv1Hz',
            
            'ca30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/ca30Hz',
            'ca3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/ca3Hz',
            'ca1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/ca1Hz',

            'hivt30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/hivt30Hz',
            'hivt3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/hivt3Hz',
            'hivt1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/hivt1Hz',

            'lanegcn30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lanegcn30Hz',
            'lanegcn3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lanegcn3Hz',
            'lanegcn1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lanegcn1Hz',
            
            'knndefault30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/knndefault30Hz',
            'knndefault3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/knndefault3Hz',
            'knndefault1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/knndefault1Hz',
            
            'knnsocial30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/knnsocial30Hz',
            'knnsocial3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/knnsocial3Hz',
            'knnsocial1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/knnsocial1Hz',
            
            'lstmdefault30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lstmdefault30Hz',
            'lstmdefault3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lstmdefault3Hz',
            'lstmdefault1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lstmdefault1Hz',
        
            'lstmsocial30Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lstmsocial30Hz',
            'lstmsocial3Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lstmsocial3Hz',
            'lstmsocial1Hz': '/home/phong/driving_data/official/despot_planner/smu_server/lstmsocial1Hz',  
        }

    prediction_performance_DESPOT = {}
    driving_performance_DESPOT = {}
    algorithm_performance_DESPOT = {}
    
    # Storing to a pickle file
    if  args.mode == "data":

        for key in directories_map_DESPOT.keys():
            print(f"Processing {key}")
            prediction_performance, driving_performance, tree_performance = get_dynamic_ade(directories_map_DESPOT[key])
            prediction_performance_DESPOT[key] = prediction_performance
            driving_performance_DESPOT[key] = driving_performance
            algorithm_performance_DESPOT[key] = tree_performance

            save_dict_to_file(prediction_performance_DESPOT, f'pickle_files2/prediction_performance_DESPOT_sameHz_{key}.pickle')
            save_dict_to_file(driving_performance_DESPOT, f'pickle_files2/driving_performance_DESPOT_sameHz_{key}.pickle')
            save_dict_to_file(algorithm_performance_DESPOT, f'pickle_files2/algorithm_performance_DESPOT_sameHz_{key}.pickle')
            print(f"Finish saving to pickle file. Exit")
        
        exit(0)

    elif args.mode == "plot":
        for key in directories_map_DESPOT.keys():
            prediction_performance_DESPOT[key] = load_dict_from_file(f'pickle_files/prediction_performance_DESPOT_sameHz_{key}.pickle')[key]
            driving_performance_DESPOT[key] = load_dict_from_file(f'pickle_files/driving_performance_DESPOT_sameHz_{key}.pickle')[key]
            algorithm_performance_DESPOT[key] = load_dict_from_file(f'pickle_files/algorithm_performance_DESPOT_sameHz_{key}.pickle')[key]
    else:
        assert False, f"Not available option form {args.mode}"    

    methods = list(directories_map_DESPOT.keys())
    #performance_summit = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_DESPOT = np.zeros(len(methods))
    performance_DESPOT_obs20 = np.zeros(len(methods))
    performance_DESPOT_closest = np.zeros(len(methods))
    
    # Printing out tree performance
    # for method in methods:
    #     for metric in algorithm_performance_DESPOT[method].keys():
    #         values =  algorithm_performance_DESPOT[method][metric]
    #         if isinstance(values, int):
    #             print(f"Method {method} Metric {metric} with Value: {values:.2f}")
    #         elif isinstance(values, list) and len(values) == 0:
    #             print(f'Error at {method} {metric}')
    #         elif isinstance(values[0], list):
    #             flattened_list = [value for sublist in values for value in sublist]
    #             print(f"Method {method} Metric {metric} with Average Value: {np.mean(np.array(flattened_list)):.2f}")
    #         else:
    #             print(f"Method {method} Metric {metric} with Average Value: {np.mean(np.array(values)):.2f}")
    
    # for key in prediction_performance_DESPOT.keys():
    #     idx = methods.index(key)
    #     performance_DESPOT[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade'])
    #     performance_DESPOT_obs20[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade_obs20'])
    #     performance_DESPOT_closest[idx] = np.nanmean(prediction_performance_DESPOT[key]['ade_closest'])
    #     print(f"DESPOT Mean ADE for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade']):.2f} and "
    #           f"Mean ADE obs20 for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade_obs20']):.2f} and "
    #           f"Mean ADE closest for {key} is {np.nanmean(prediction_performance_DESPOT[key]['ade_closest']):.2f}")

    #method_performance = get_method_performance(prediction_performance_DESPOT, driving_performance_DESPOT, methods, 'ade')

    # Section II, 3rd Drawbacks: does not care time constraints
    # Plot 1 - Only Either Driving Performance, Efficiency, Safety, Comfort in 1 plot. 
    # Bar plot. Each sub-bar is method, each bar in sub-bar is frequency. Each bar is mean value
    #plot_driving_performance_sameHz(prediction_performance_DESPOT, driving_performance_DESPOT, list(prediction_performance_DESPOT.keys()))
    # We can also have scatter plot here. 

    # Plot 2 - Threshold
    # A line plot . Horizontal is only Hz, vertical is driving performance. Each line is each method
    # We see the saturated points
    plot_driving_performance_sameHz(prediction_performance_DESPOT, driving_performance_DESPOT, 
                                           algorithm_performance_DESPOT, list(prediction_performance_DESPOT.keys()))

    exit(0) # Exit here as we do not need prediction performance.

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



