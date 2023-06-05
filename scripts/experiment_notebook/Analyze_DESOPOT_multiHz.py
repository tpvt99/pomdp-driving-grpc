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

from scipy import stats
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r'\usepackage{newtxtext,newtxmath}'
from Analyze_DESPOT import find_max_min, get_normalized_data

from common_performance_stats import get_dynamic_ade

# Weights for each metric within a category
safety_weights = {
    'collision_rate': 0.0,
    'near_miss_rate': 1.0,
    'near_distance_rate': 0.0,
    'mean_min_ttc': 0.0,
    'mean_min_ttr': 0.0
}
safety_directions = {
    'collision_rate': 'lower',
    'near_miss_rate': 'lower',
    'near_distance_rate': 'lower',
    'mean_min_ttc': 'higher',
    'mean_min_ttr': 'higher'
}
comfort_weights = {
    'jerk': 1.0,
    'lateral_acceleration': 0.0,
    'acceleration': 0.0
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


def plot_best_fit_line(x, y, ax, color="blue"):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate the best-fit line
    line = slope * np.array(x) + intercept

    # Plot the best-fit line
    ax.plot(x, line, color=color, alpha=1.0)
    print(r_value)

    # Add regression statistics to the plot
    #stats_text = f"""\
    #R^2: {r_value ** 2:.2f}
    #p-value: {p_value:.4f}
    #"""
    #ax.text(0.05, 0.15, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
    #        color='black')

    return ax


def plot_driving_performance_sameHz(prediction_performance_map, driving_performance_map, 
                                    algorithm_performance_DESPOT,
                                    methods_to_plot, driving_metric='driving'):
    # methods to plot is like [cv30Hz, cv1Hz, cv3Hz]
    # Remap for unified name of plotting througout the papers
    #3print(prediction_performance_map['knndefault30Hz'])
    frequencies = ['30Hz', '3Hz', '1Hz']
    remap_methods = {
        'cv': 'CV',
        'ca': 'CA',
        'knndefault': 'KNN',
        'knnsocial': 'S-KNN',
        'hivt': 'HiVT',
        'lanegcn': 'LaneGCN',
        'lstmdefault': 'LSTM',
        'lstmsocial': 'S-LSTM'
    }
    # for k, v in remap_methods.items():
    #     for freq in frequencies:
    #         prediction_performance_map[f"{v}{freq}"] = prediction_performance_map[f"{k}{freq}"]
    #         del prediction_performance_map[f"{k}{freq}"]

    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])
    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])
    # Using static ADE

    static_metric = 'FDE'
    if static_metric == "ADE":
        performance_summit = performance_summit_ade
    elif static_metric == "FDE":
        performance_summit = performance_summit_fde
    elif static_metric == "minADE":
        performance_summit = performance_summit_made
    elif static_metric == "minFDE":
        performance_summit = performance_summit_mfde

    dynamic_metric = 'fde_predlen30'
    method_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                     methods_to_plot, dynamic_metric)
    driving_performance = {method: method_performance[method]['driving'] for method in methods_to_plot}
    print(f"driving: {driving_performance}")

    performance_mean = {}
    #print(prediction_performance_DESPOT.keys())
    #for method in methods_to_plot:
    #    # Cannot use ade30 for 3Hz as cannot have enough values
    #    performance_mean[method] = np.nanmean(prediction_performance_DESPOT[method]['ade'])

    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_colors = ['#8c564b','#ff7f0e', '#2ca02c']
    index = np.arange(len(methods_to_plot))
    markers = ['o', 's', 'D', 'v', '^', '>', '<', 'p']  # List of markers for each method

    # Plot combined frequencies
    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots(1,1, figsize=(5,5))
    methods_to_plot_strippping_freq = [
        'cv',
        'ca',
         'knndefault',
         'knnsocial',
         'hivt',
         'lanegcn',
         'lstmdefault',
         'lstmsocial']

    # plot_type = 'static_ade_vs_driving_performance'
    plot_type = 'dynamic_ade_vs_driving_performance'
    if plot_type == 'static_ade_vs_driving_performance':
        sorted_indices = np.argsort(performance_summit)
        performance_summit_sorted = performance_summit[sorted_indices]
        methods_to_plot_strippping_freq = np.array(methods_to_plot_strippping_freq)[sorted_indices]

        # If plotting Static ADE vs Dynamic ADE, using old performance_mean
        #performance_mean = performance_mean
        # Else Static ADE vs Driving performance, using new one
        performance_mean = driving_performance

        for j, freq in enumerate(frequencies):
            performance = []

            for i, method in enumerate(methods_to_plot_strippping_freq):
                assert f"{method}{freq}" in performance_mean.keys()
                if np.isnan(performance_mean[f"{method}{freq}"]):
                    performance.append(0.0)
                else:
                    performance.append(performance_mean[f"{method}{freq}"])
            print(performance)
            freq_text = freq
            ax1.plot(performance_summit_sorted, performance, label=f"{freq_text}", color=default_colors[j % len(default_colors)], 
                    marker=markers[j], markersize = 5)


        plt.title('', fontsize=16)
        ax1.set_xlabel(f'Static {static_metric}', fontsize=16)
        ax1.set_ylabel(f'Driving Performance', fontsize=16)
        #ax1.set_title('Static vs. Dynamic ADE at Different Planning Frequencies')
        #ax1.set_xticks(performance_summit_sorted)
        #ax1.set_xticklabels(methods_to_plot_strippping_freq)
        #ax1.legend(ncol=1, loc='lower left', bbox_to_anchor=(0.5, 0.1), frameon=True)
        ax1.legend()
        #ax1.set_ylim([0.3, 0.9])
        plt.tight_layout()
        ax1.grid(True, linewidth=0.5, alpha=0.7)
        # Add this block after the for loop of frequencies (after the line with ax1.grid(True, linewidth=0.5, alpha=0.7))
        #for i, (x_value, method) in enumerate(zip(performance_summit_sorted, methods_to_plot_strippping_freq)):
        #    ax1.axvline(x=x_value, ymin=0, ymax=1, linestyle="--", color="gray", alpha=0.5)
        #    ax1.text(x_value, ax1.get_ylim()[0], method, rotation=90, ha="right", va="bottom")


        # If plotting Static ADE vs Dynamic ADE, using old performance_mean
        #performance_mean = performance_mean
        # Else Static ADE vs Driving performance, using new one

        # Display the legend horizontally using ncol parameter
        #ax1.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.15))

        plt.savefig(f"C_DESPOT_sameHz_static_{static_metric}_driving_performance.pdf", dpi=300)
        plt.clf()
    else:
        fig, ax = plt.subplots(1,1,figsize=(5, 5)) # adjust figure size as needed
        plt.rcParams.update({'font.size': 16})

        dynamic_metric = 'fde_obs20_closest_predlen30'
        x_values_for_plot = []
        y_values_for_plot = []

        for j, freq in enumerate(frequencies):
            performance = []

            freq_text = freq

            x_values_for_plot_1 = []
            y_values_for_plot_1 = []

            for i, method in enumerate(methods_to_plot_strippping_freq):
                prediction_data = np.nanmean(np.array(prediction_performance_map[f"{method}{freq}"][dynamic_metric]))
                if np.isnan(prediction_data):
                    continue
                x_values_for_plot.append(prediction_data)
                y_values_for_plot.append(driving_performance[f"{method}{freq}"])
                x_values_for_plot_1.append(prediction_data)
                y_values_for_plot_1.append(driving_performance[f"{method}{freq}"])
        
            ax.scatter(x_values_for_plot_1, y_values_for_plot_1, label=f"{freq_text}", color=default_colors[j % len(default_colors)])

        ax = plot_best_fit_line(x_values_for_plot, y_values_for_plot, ax)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values_for_plot, y_values_for_plot)
        plt.title('$\mathrm{R}^2$' + f': {r_value **2:.2f} $\;$ $p$-value: {p_value:.3f}', fontsize=16)
        ax.set_xlabel('Dynamic ADE', fontsize=16)
        ax.set_ylabel('Driving Performance', fontsize=16)
        #ax1.set_title('Static vs. Dynamic ADE at Different Planning Frequencies')
        #ax1.set_xticks(performance_summit_sorted)
        #ax1.set_xticklabels(methods_to_plot_strippping_freq)
        ax.legend(loc=3)
        #ax1.set_ylim([0.3, 0.9])
        plt.tight_layout()
        ax.grid(True, linewidth=0.5, alpha=0.7)

        # Display the legend horizontally using ncol parameter
        #ax1.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.15))

        plt.savefig(f"B_DESPOT_dynamic_{dynamic_metric.upper()}_driving_performance.pdf", dpi=300)
        plt.clf()



def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = {}
    for i, method in enumerate(methods_to_plot):
        _, safety_data, comfort_data, efficiency_data, performance = get_normalized_data(method, prediction_performance, 
                        driving_performance, max_min_method, prediction_metric)

        method_performance[method] = {'safety': np.nanmean(safety_data), 
                                      'comfort': np.nanmean(comfort_data),
                                 'efficiency': np.nanmean(efficiency_data), 
                                 'driving': np.nanmean(performance)}
        
        method_performance[i] = np.nanmean(performance)

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

            save_dict_to_file(prediction_performance_DESPOT, f'pickle_filesdespot_sameHz/prediction_performance_DESPOT_sameHz_{key}.pickle')
            save_dict_to_file(driving_performance_DESPOT, f'pickle_filesdespot_sameHz/driving_performance_DESPOT_sameHz_{key}.pickle')
            save_dict_to_file(algorithm_performance_DESPOT, f'pickle_filesdespot_sameHz/algorithm_performance_DESPOT_sameHz_{key}.pickle')
            print(f"Finish saving to pickle file. Exit")
        
        exit(0)

    elif args.mode == "plot":
        for key in directories_map_DESPOT.keys():
            prediction_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot_sameHz/prediction_performance_DESPOT_sameHz_{key}.pickle')[key]
            driving_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot_sameHz/driving_performance_DESPOT_sameHz_{key}.pickle')[key]
            algorithm_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot_sameHz/algorithm_performance_DESPOT_sameHz_{key}.pickle')[key]
    else:
        assert False, f"Not available option form {args.mode}"    

    methods = list(directories_map_DESPOT.keys())
    #performance_summit = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_DESPOT = np.zeros(len(methods))
    performance_DESPOT_obs20 = np.zeros(len(methods))
    performance_DESPOT_closest = np.zeros(len(methods))
    
    # Prin
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
