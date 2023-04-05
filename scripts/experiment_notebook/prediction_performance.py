import numpy as np

def displacement_error(pred, gt):
    return np.linalg.norm(np.array(pred) - np.array(gt))

def ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list):
    ego_ade = []
    ego_fde = []
    ego_obs_list = {} # exo observation

    # Calculate ADE and FDE for ego agent
    for i, timestep in enumerate(pred_car_list):

        # Skip the first 20 timesteps because the prediction is not accurate
        if timestep < 0: # Change to 20 to have prev pic. I put 0 to have dynamic
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
        if timestep < 0: # I put 0 to have dynamic
            continue

        for agent_index, exo_agent in enumerate(exos_list[timestep]):

            # if agent_index != 0:
            #     continue

            # For safety analysis, limit 2 things
            # 1/ Prediction len = 5 as recent time affects much more to safety and jerky
            

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

                if timestep >= 20:
                    all_agent_ids_at_prev_20_timestep = [agent['id'] for agent in exos_list[timestep+j-20]]
                
                # If agent is not in the next timestep, then there is no point in calculating the error
                if agent_id not in all_agent_ids_at_next_timestep:
                    break
                
                # If agent not in prev 20 timestep, then it not have enough history to calculate the error
                #if agent_id not in all_agent_ids_at_prev_20_timestep:
                #    break
                
                if  timestep >= 20 and j < 20 and agent_id in all_agent_ids_at_prev_20_timestep:
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
