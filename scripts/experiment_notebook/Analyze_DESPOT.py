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
from matplotlib import colors
from scipy import stats

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



def plot_dynamic_vs_staticade(prediction_performance_map, methods_to_plot):

    methods_to_plot = ['cv', 'ca', 'knndefault', 'knnsocial', 'hivt', 'lanegcn', 'lstmdefault', 'lstmsocial']
    performance_summit = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_RVO = np.zeros(len(methods_to_plot))
    performance_RVO_predlen30 = np.zeros(len(methods_to_plot))
    performance_RVO_obs20 = np.zeros(len(methods_to_plot))
    performance_RVO_closest = np.zeros(len(methods_to_plot))
    performance_RVO_20meters_closest = np.zeros(len(methods_to_plot))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Printing out tree performance
    for key in prediction_performance_map.keys():
        idx = methods_to_plot.index(key)
        performance_RVO[idx] = np.nanmean(prediction_performance_map[key]['ade'])
        performance_RVO_predlen30[idx] = np.nanmean(prediction_performance_map[key]['ade_predlen30'])
        performance_RVO_obs20[idx] = np.nanmean(prediction_performance_map[key]['ade_obs20'])
        performance_RVO_closest[idx] = np.nanmean(prediction_performance_map[key]['ade_closest'])
        performance_RVO_20meters_closest[idx] = np.nanmean(prediction_performance_map[key]['ade_20meters_closest'])
        print(f"Mean ADE {key} is {np.nanmean(prediction_performance_map[key]['ade']):.2f} "
              f"ADE ADE predlen30 {key} is {np.nanmean(prediction_performance_map[key]['ade_predlen30']):.2f} " 
              f"ADE obs20 {key} is {np.nanmean(prediction_performance_map[key]['ade_obs20']):.2f}"
              f"ADE closest {key} is {np.nanmean(prediction_performance_map[key]['ade_closest']):.2f}"
              f"ADE closest 20meters {key} is {np.nanmean(prediction_performance_map[key]['ade_20meters_closest']):.2f}")
   

    offsets=0.4
    space_between_methods = 1.2
    position = np.arange(len(methods_to_plot)) * (len(methods_to_plot) * offsets / 2 + space_between_methods)

    wrapped_methods = []
    for method in methods_to_plot:
        if 'knn' in method.lower():
            wrapped_methods.append(method.replace('knn', 'knn\n', 1))
        elif 'lstm' in method.lower():
            wrapped_methods.append(method.replace('lstm', 'lstm\n', 1))
        else:
            wrapped_methods.append(textwrap.fill(method, 12))

    fig, ax = plt.subplots(figsize=(8.5,5))
    #fig.set_size_inches(18, 9)
    ax.bar(position-2*offsets, performance_summit, width=offsets, color=default_colors[0], align='center', label='Summit_dataset (S)')
    ax.bar(position-1*offsets, performance_RVO_predlen30, width=offsets, color=(*colors.to_rgba(default_colors[1])[:3], 0.5), align='center', label='ADE_pred30 (D)')
    ax.bar(position+0*offsets, performance_RVO_closest, width=offsets, color=(*colors.to_rgba(default_colors[2])[:3], 0.5), align='center', label='ADE_closest (D)')
    ax.bar(position+1*offsets, performance_RVO, width=offsets, color=(*colors.to_rgba(default_colors[3])[:3], 0.5), align='center', label='ADE (D)')
    ax.bar(position+2*offsets, performance_RVO_obs20, width=offsets, color=(*colors.to_rgba(default_colors[4])[:3], 0.5), align='center', label='ADE_obs20 (D)')
    #ax.bar(position+2*offsets, performance_RVO_20meters_closest, width=offsets, color=default_colors[4], align='center', label='RVO_20meters')
    
    ax.set_xticks(position)
    ax.set_xticklabels(wrapped_methods)
    ax.set_title('Comparison of Static_ADE vs. Dynamic_ADE of DESPOT')
    ax.set_ylabel('ADE')
    #ax.legend(loc='upper right', fontsize='small')
    ax.legend(ncol=5, loc='upper center', fontsize='small')
    ax.set_ylim(0, 5)  # Adjust the ylim to provide more space for the legend

    plt.tight_layout()
    plt.savefig("3_Static_ADE_vs_Dynamic_ADE_DESPOT.png")
    plt.clf()

def plot_best_fit_line(x, y, ax, color="blue"):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate the best-fit line
    line = slope * np.array(x) + intercept

    # Plot the best-fit line
    ax.plot(x, line, color=color, label='Best-fit line', alpha=0.1)

    # Calculate the 95% confidence interval
    #t = stats.t.ppf(1 - 0.025, len(x) - 2)  # 95% confidence level
    #slope_ci = t * std_err
    #intercept_ci = t * (std_err * np.sqrt(np.sum(x**2) / len(x)))


    # Calculate the endpoints for the confidence interval lines
    #lower_line = (slope - slope_ci) * np.array(x) + (intercept - intercept_ci)
    #upper_line = (slope + slope_ci) * np.array(x) + (intercept + intercept_ci)


    # Plot the confidence interval lines
    #ax.plot(x, lower_line, color=color, linestyle='--', alpha=0.7, label='Lower 95% CI')
    #ax.plot(x, upper_line, color=color, linestyle='--', alpha=0.7, label='Upper 95% CI')

    # Add regression statistics to the plot
    stats_text = f"""\
    R^2: {r_value ** 2:.2f}
    p-value: {p_value:.4f}
    Std Error: {std_err:.4f}
    """
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

    return ax

def plot_static_ade_vs_driving_performance(prediction_performance_map, 
                                                   driving_performance_map, metric='FDE'):

    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])

    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])

    if metric == "ADE":
        performance_summit = performance_summit_ade
    elif metric == "FDE":
        performance_summit = performance_summit_fde
    elif metric == "minADE":
        performance_summit = performance_summit_made
    elif metric == "minFDE":
        performance_summit = performance_summit_mfde

    methods_to_plot = ['cv', 'ca', 'knndefault', 'knnsocial', 'hivt', 'lanegcn', 'lstmdefault', 'lstmsocial']
    
    method_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                     methods_to_plot, "ade")

    ## Static ADE vs Driving Performance##


    # Create a scatter plot with different colors for each point
    colors = plt.cm.rainbow(np.linspace(0, 1, len(methods_to_plot)))
    fig, ax = plt.subplots(figsize=(5, 5)) # adjust figure size as needed
    print(method_performance)
    for i, method in enumerate(methods_to_plot):
        plt.scatter(performance_summit[i], method_performance[i], color=colors[i], label=method)


        plt.annotate(method, (performance_summit[i], method_performance[i]), 
                     textcoords="offset points", xytext=(-2,7), ha='center')

    # Add labels and legend
    plt.xlabel(f'Static {metric}')
    plt.ylabel('Driving Performance')
    #plt.legend()
    plt.ylim(top=max(method_performance)*1.01)  # Adjust the ylim to provide more space for the legend
    plt.xlim(right=max(performance_summit)*1.05)
    plt.title(f'Relation of Static {metric} vs Driving Performance \n for DESPOT Planner')
    
    ax = plot_best_fit_line(performance_summit, method_performance, ax)

    plt.tight_layout()

    plt.savefig(f"4_Static_{metric}_vs_Driving_Performance_DESPOT.png")
    plt.clf()

def plot_dynamic_ade_vs_driving_performance(prediction_performance_map, 
                                            driving_performance_map, metric='minFDE'):
    
    methods_to_plot = ['cv', 'ca', 'knndefault', 'knnsocial', 'hivt', 'lanegcn', 'lstmdefault', 'lstmsocial']
    #methods_to_plot = ['cv', 'ca',  'hivt', 'lstmdefault', 'lstmsocial']
    r_values_dict = {}

    # Find the best-fit-line
    for best_fit_metric in prediction_performance_map['cv'].keys():
        if "ade" not in best_fit_metric and "fde" not in best_fit_metric:
            continue
        driving_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                        methods_to_plot, best_fit_metric)
        dynamic_metric = [np.nanmean(prediction_performance_map[key][best_fit_metric]) for key in methods_to_plot]

        x = dynamic_metric
        y = driving_performance

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Store the r-value in the dictionary
        r_values_dict[best_fit_metric] = r_value

        # Print the r-value for the current method
        print(f"Method: {best_fit_metric}, R-value: {r_value} p-value {p_value}")

    # Sort the methods based on their r-values (ascending order)
    sorted_r_values = sorted(r_values_dict.items(), key=lambda x: x[1], reverse=False)

    # Print the best, 2nd best, and worst methods based on r-values
    for i in range(10):
        print(f"{i}th method: {sorted_r_values[i][0]}, R-value: {sorted_r_values[i][1]}")
    print(f"Worst method: {sorted_r_values[-1][0]}, R-value: {sorted_r_values[-1][1]}")

    
    ## Dynamic ADE vs Driving Performance##
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    metric = 'ade_obs20_20meters_closest_predlen30'
    driving_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                        methods_to_plot, 'ade')
    dynamic_metric = [np.nanmean(prediction_performance_map[key][metric]) for key in methods_to_plot]
   
    # Create a scatter plot with different colors for each point
    colors = plt.cm.rainbow(np.linspace(0, 1, len(methods_to_plot)))
    fig, ax = plt.subplots(figsize=(5, 5)) # adjust figure size as needed

    for i, method in enumerate(methods_to_plot):
        plt.scatter(dynamic_metric[i], driving_performance[i], color=colors[i], label=method)
        plt.annotate(method, (dynamic_metric[i], driving_performance[i]), 
                     textcoords="offset points", xytext=(-2,7), ha='center')
    print('xx')

    # Add labels and legend
    plt.xlabel(f'Dynamic {metric}')
    plt.ylabel('Driving Performance')
    plt.ylim(top=max(driving_performance)*1.01)  # Adjust the ylim to provide more space for the legend
    plt.xlim(right=max(dynamic_metric)*1.06)
    #plt.xlim(left  = min(min(performance_summit_ade), min(performance_summit_fde), min(performance_summit_made), min(performance_summit_mfde))*0.95,
    #         right = max(max(performance_summit_ade), max(performance_summit_fde), max(performance_summit_made), max(performance_summit_mfde))*1.05)
    plt.title(f'Relation of Dynamic {metric} \n vs Driving Performance of DESPOT Planner')

    ax = plot_best_fit_line(dynamic_metric, driving_performance, ax)

    plt.tight_layout()
    
    plt.savefig(f"5_Dynamic_{metric}_vs_Driving_Performance_DESPOT.png")
    plt.clf()

def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = np.zeros(len(methods_to_plot))
    for i, method in enumerate(methods_to_plot):
        _, _, _, _, performance = get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric)
        performance = np.nan_to_num(performance)
        method_performance[i] = np.mean(performance)

    return method_performance

if __name__ == "__main__":

    args = parse_args()

    directories_map_DESPOT = {
            'cv': '/home/phong/driving_data/official/despot_planner/same_computation/cv2Hz',
            'ca': '/home/phong/driving_data/official/despot_planner/same_computation/ca2Hz',
            'hivt': '/home/phong/driving_data/official/despot_planner/same_computation/hivt02Hz',
            'lanegcn': '/home/phong/driving_data/official/despot_planner/same_computation/lanegcn02Hz',
            'lstmdefault': '/home/phong/driving_data/official/despot_planner/same_computation/lstmdefault05Hz',
            'lstmsocial': '/home/phong/driving_data/official/despot_planner/same_computation/lstmsocial03Hz/',
            'knnsocial': '/home/phong/driving_data/official/despot_planner/same_computation/knnsocial005Hz/',
            'knndefault': '/home/phong/driving_data/official/despot_planner/same_computation/knndefault005Hz/',
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

            save_dict_to_file(prediction_performance_DESPOT, f'pickle_filesdespot1/prediction_performance_DESPOT_{key}.pickle')
            save_dict_to_file(driving_performance_DESPOT, f'pickle_filesdespot1/driving_performance_DESPOT_{key}.pickle')
            save_dict_to_file(algorithm_performance_DESPOT, f'pickle_filesdespot1/algorithm_performance_DESPOT_{key}.pickle')
            print(f"Finish saving to pickle file. Exit")
        
        exit(0)

    elif args.mode == "plot":
        for key in directories_map_DESPOT.keys():
            prediction_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot1/prediction_performance_DESPOT_{key}.pickle')[key]
            driving_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot1/driving_performance_DESPOT_{key}.pickle')[key]
            algorithm_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot1/algorithm_performance_DESPOT_{key}.pickle')[key]
    else:
        assert False, f"Not available option form {args.mode}"

    
    for method in list(directories_map_DESPOT.keys()):
        for metric in algorithm_performance_DESPOT[method].keys():
                values = algorithm_performance_DESPOT[method][metric]
                if isinstance(values, int):
                    print(f"Method {method} Metric {metric} with Value: {values:.4f}")
                elif isinstance(values, list) and len(values) == 0:
                    print(f'Error at {method} {metric}')
                elif isinstance(values[0], list):
                    flattened_list = [value for sublist in values for value in sublist]
                    print(f"Method {method} Metric {metric} with Average Value: {np.mean(np.array(flattened_list)):.4f}")
                else:
                    print(f"Method {method} Metric {metric} with Average Value: {np.mean(np.array(values)):.4f}")
                # if metric == "number_timesteps":
                #     for i in range(len(values)-1,-1,-1):
                #         if values[i] < 1:
                #             del driving_performance_DESPOT[method]['safety']['collision_rate'][i]
                #             del driving_performance_DESPOT[method]['safety']['near_miss_rate'][i]
                #             del driving_performance_DESPOT[method]['safety']['near_distance_rate'][i]
                #             del driving_performance_DESPOT[method]['safety']['mean_min_ttc'][i]
                #             del driving_performance_DESPOT[method]['safety']['mean_min_ttr'][i]
                #             del driving_performance_DESPOT[method]['comfort']['jerk'][i]
                #             del driving_performance_DESPOT[method]['comfort']['lateral_acceleration'][i]
                #             del driving_performance_DESPOT[method]['comfort']['acceleration'][i]
                #             del driving_performance_DESPOT[method]['efficiency']['avg_speed'][i]
                #             del driving_performance_DESPOT[method]['efficiency']['tracking_error'][i]
                #             del driving_performance_DESPOT[method]['efficiency']['efficiency_time'][i]
                #             del driving_performance_DESPOT[method]['efficiency']['distance_traveled'][i]
                        # for pred_metric in prediction_performance_DESPOT[method].keys():
                        #     del prediction_performance_DESPOT[method][pred_metric][i]

    
    #plot_dynamic_vs_staticade(prediction_performance_DESPOT, list(directories_map_DESPOT.keys()))

    #plot_static_ade_vs_driving_performance(prediction_performance_DESPOT, driving_performance_DESPOT)

    plot_dynamic_ade_vs_driving_performance(prediction_performance_DESPOT, driving_performance_DESPOT)

    exit(0) # Exit here as we do not need prediction performance.

    ## plot static ade vs dynamic ade ##
    

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



