import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys
from average_statistics_of_1_data import merge_3_get_statistics

# This file show statistics of search tree. This gives deeper understanding about the search.
# And whether the implementation is correct

FOLDER_PATHS = [#'/home/cunjun/driving_data_benchmark/cv1/result/joint_pomdp_drive_mode/',
                #'/home/cunjun/driving_data_benchmark/ca1/result/joint_pomdp_drive_mode/',
                #'/home/cunjun/driving_data_benchmark/gamma1/result/joint_pomdp_drive_mode/',
                #'/home/cunjun/driving_data_benchmark/gamma1_update/result/joint_pomdp_drive_mode/',
                #'/home/cunjun/driving_data_benchmark/gamma2/result/joint_pomdp_drive_mode/',
                #'/home/cunjun/driving_data_benchmark/gamma3/result/joint_pomdp_drive_mode/'
]

FOLDER_PATHS = ['/home/cunjun/driving_data/representative/original_gamma_nosync/',
            #'/home/cunjun/driving_data/representative/gamma_tick_3Hz_change_allHz/',
            #'/home/cunjun/driving_data/representative/gamma_tick_30Hz_change_allHz/',
           #'/home/cunjun/driving_data/representative/gamma_tick_3Hz_no_changeHz/',
            #'/home/cunjun/driving_data/representative/gamma_tick_10Hz_change_allHz/',
            '/home/cunjun/driving_data/representative/gamma_tick_30Hz_timescale1_planner_8e3/',
            '/home/cunjun/driving_data/representative/gamma_tick_30Hz_timescale1_planner_8e4/',
            '/home/cunjun/driving_data/representative/gamma_tick_30Hz_timescale1_planner_8e5/',

]

if __name__ == "__main__":

    # Internal DESPOT
    average_moped_steps = []
    average_time_each_moped_step = []
    average_tree_depth = []
    average_actions_per_scenario = []
    average_trials_per_tree_search = []
    average_action_execution_steps_per_scenario = []
    average_time_each_action_search = []
    average_node_expansions = []

    # External results
    average_deceleration = []
    average_acceleration = []
    average_maintain = []
    average_col_rate = []
    average_travel_distance = []
    average_rewards = []
    average_progress = []
    average_stuck = []

    # Exo-agents related and time sync

    fig, axs = plt.subplots(4, 4, figsize=(20,20)) # we plot 4 values so we need 2 times 2. Can be adjusted if draw more than that

    count = 0
    labels = []
    for folder_path in FOLDER_PATHS:
        (total_steps, total_times, total_trials, total_node_expansions, total_node_in_total, tree_depth, prediction_time,
            total_execution_steps, total_running_time_each_scenario), \
        (connect_finish_at, time_car_states, time_agent_arrays, agent_counts, total_agents, agent_steps_counts, agent_speeds), \
            (dec_np, acc_np, mat_np, col_rate, trav_np, rew_np, progress_np, ave_speeds_np)= merge_3_get_statistics(folder_path)

        experiment_name = folder_path.split('/')[5]
        labels.append(experiment_name)

        print('-----------------------------')
        print(f"Experiment {experiment_name}")
        print(f" Total Motion Prediction Steps: and avg: {np.mean(np.array(total_steps))}")
        print(f" Time for each Search (One Tree Construction):and avg: {np.mean(np.array(total_times))}")
        print(f" Max tree depth each Search (One Tree Construction):  and avg: {np.mean(np.array(tree_depth))}")
        print(f" Trials: {np.mean(np.array(total_trials))}")
        print(f" Node expansions: {np.mean(np.array(total_node_expansions))}")
        print(f" Total of Node: {np.mean(np.array(total_node_in_total))}")
        print(f" Pred time average: {np.mean(np.array(prediction_time))} ")
        print(f" Action Execution steps average: {np.mean(np.array(total_execution_steps))} ")
        print(f" Running time of scenario average: {np.mean(np.array(total_running_time_each_scenario))} ")


        average_moped_steps.append(np.mean(np.array(total_steps)))
        average_time_each_moped_step.append(np.mean(np.array(prediction_time)))
        average_tree_depth.append(np.mean(np.array(tree_depth)))
        average_actions_per_scenario.append(np.mean(np.array(total_execution_steps)))
        average_trials_per_tree_search.append(np.mean(np.array(total_trials)))
        average_action_execution_steps_per_scenario.append(np.mean(np.array(total_running_time_each_scenario)))
        average_time_each_action_search.append(np.mean(np.array(total_times)))
        average_node_expansions.append(np.mean(np.array(total_node_expansions)))

        average_deceleration.append(np.mean(dec_np))
        average_acceleration.append(np.mean(acc_np))
        average_maintain.append(np.mean(mat_np))
        average_col_rate.append(np.mean(col_rate))
        average_travel_distance.append(np.mean(trav_np))
        average_rewards.append(np.mean(rew_np))
        average_progress.append(np.mean(progress_np))
        average_stuck.append(np.mean(ave_speeds_np))

        count += 1

    # labels = [
    #     "constant vel",
    #     "constant acc",
    #     "original gamma",
    #     "gamma1 xxx",
    #     "gamma1 slowdown 10",
    #     "gamma2 slowdown 100"
    # ]

    for i in range(count):
        axs[0][0].bar(i, average_moped_steps[i], label=labels[i])
        axs[0][0].text(x=i, y= average_moped_steps[i] * 1.05,
                       s=round(float(average_moped_steps[i]),2), ha="center")

        axs[0][1].bar(i, average_time_each_moped_step[i], label=labels[i])
        axs[0][1].text(x=i, y= average_time_each_moped_step[i] * 1.05,
                       s=round(float(average_time_each_moped_step[i]),6), ha="center")

        axs[0][2].bar(i, average_tree_depth[i], label=labels[i])
        axs[0][2].text(x=i, y= average_tree_depth[i] * 1.05,
                       s=round(float(average_tree_depth[i]), 2), ha="center")

        axs[0][3].bar(i, average_trials_per_tree_search[i], label=labels[i])
        axs[0][3].text(x=i, y= average_trials_per_tree_search[i] * 1.05,
                       s=round(float(average_trials_per_tree_search[i]),2), ha="center")

        axs[1][0].bar(i, average_actions_per_scenario[i], label=labels[i])
        axs[1][0].text(x=i, y= average_actions_per_scenario[i] * 1.05,
                       s=round(float(average_actions_per_scenario[i]),2), ha="center")

        axs[1][1].bar(i, average_action_execution_steps_per_scenario[i], label=labels[i])
        axs[1][1].text(x=i, y= average_action_execution_steps_per_scenario[i] * 1.05,
                       s=round(float(average_action_execution_steps_per_scenario[i]),2), ha="center")

        axs[1][2].bar(i, average_time_each_action_search[i], label=labels[i])
        axs[1][2].text(x=i, y= average_time_each_action_search[i] * 1.05,
                   s=round(float(average_time_each_action_search[i]),2), ha="center")

        axs[1][3].bar(i, average_node_expansions[i], label=labels[i])
        axs[1][3].text(x=i, y= average_node_expansions[i] * 1.05,
                   s=round(float(average_node_expansions[i]),2), ha="center")

        axs[2][0].bar(i, average_col_rate[i], label=labels[i])
        axs[2][0].text(x=i, y= average_col_rate[i] * 1.05,
                       s=round(float(average_col_rate[i]),2), ha="center")

        axs[2][1].bar(i, average_travel_distance[i], label=labels[i])
        axs[2][1].text(x=i, y= average_travel_distance[i] * 1.05,
                   s=round(float(average_travel_distance[i]),2), ha="center")

        axs[2][2].bar(i, average_rewards[i], label=labels[i])
        axs[2][2].text(x=i, y= average_rewards[i] * 1.05,
                       s=round(float(average_rewards[i]),2), ha="center")

        axs[2][3].bar(i, average_progress[i], label=labels[i])
        axs[2][3].text(x=i, y= average_progress[i] * 1.05,
                       s=round(float(average_progress[i]),2), ha="center")

        axs[3][0].bar(i, average_deceleration[i], label=labels[i])
        axs[3][0].text(x=i, y= average_deceleration[i] * 1.05,
                       s=round(float(average_deceleration[i]),2), ha="center")

        axs[3][1].bar(i, average_acceleration[i], label=labels[i])
        axs[3][1].text(x=i, y= average_acceleration[i] * 1.05,
                       s=round(float(average_acceleration[i]),2), ha="center")

        axs[3][2].bar(i, average_maintain[i], label=labels[i])
        axs[3][2].text(x=i, y= average_maintain[i] * 1.05,
                       s=round(float(average_maintain[i]),2), ha="center")

        axs[3][3].bar(i, average_stuck[i], label=labels[i])
        axs[3][3].text(x=i, y= average_stuck[i] * 1.05,
                       s=round(float(average_stuck[i]),2), ha="center")

    axs[0][0].set_title("Average of total Motion Prediction Steps")
    axs[0][0].set_ylim(0, 1.2 * np.max(average_moped_steps))

    axs[0][1].set_title("Average of prediction time (per call to moped)")
    axs[0][1].set_ylim(0, 1.2 * np.max(average_time_each_moped_step))

    axs[0][2].set_title("Average of tree depth per tree construction")
    axs[0][2].set_ylim(0, 1.2 * np.max(average_tree_depth))

    axs[0][3].set_title("Average of trials per tree construction")
    axs[0][3].set_ylim(0, 1.2 * np.max(average_trials_per_tree_search))

    axs[1][0].set_title("Average of actions per scenario")
    axs[1][0].set_ylim(0, 1.2 * np.max(average_actions_per_scenario))

    axs[1][1].set_title("Average of running time per scenario")
    axs[1][1].set_ylim(0, 1.2 * np.max(average_action_execution_steps_per_scenario))

    axs[1][2].set_title("Average of time per action search")
    axs[1][2].set_ylim(0, 1.2 * np.max(average_time_each_action_search))

    axs[1][3].set_title("Average of node expansions per tree construction")
    axs[1][3].set_ylim(0, 1.2 * np.max(average_node_expansions))

    axs[2][0].set_title("Average of collision rate")
    axs[2][0].set_ylim(0, 1.2 * np.max(average_col_rate))

    axs[2][1].set_title("Average of travel distance")
    axs[2][1].set_ylim(0, 1.2 * np.max(average_travel_distance))

    axs[2][2].set_title("Average of rewards")
    axs[2][2].set_ylim(1.2 * np.min(average_rewards), 0.0)

    axs[2][3].set_title("Average of progress")
    axs[2][3].set_ylim(0, 1.2 * np.max(average_progress))

    axs[3][0].set_title("Average of deceleration rate")
    axs[3][0].set_ylim(0, 1.2 * np.max(average_deceleration))

    axs[3][1].set_title("Average of acceleration rate")
    axs[3][1].set_ylim(0, 1.2 * np.max(average_acceleration))

    axs[3][2].set_title("Average of main rate")
    axs[3][2].set_ylim(0, 1.2 * np.max(average_maintain))

    axs[3][3].set_title("Average of Speed")
    axs[3][3].set_ylim(0, 1.2 * np.max(average_stuck))


    plt.setp(axs, xticks=[])
    fig.tight_layout()
    plt.legend()
    plt.show()