import numpy as np
import copy

def smooth_l1_loss(y_true, y_pred, delta=1.0):
    assert y_true.shape == y_pred.shape, "y_true and y_pred should have the same shape"
    
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    loss = np.where(abs_error < delta, 0.5 * np.square(error), delta * (abs_error - 0.5 * delta))
    return np.mean(loss)

# Example usage:
# y_true = np.array([1.0, 2.0, 3.0])
# y_pred = np.array([1.2, 1.8, 3.5])

# loss = smooth_l1_loss(y_true, y_pred)
# print("Smooth L1 Loss:", loss)


def consistency(pred, pred_1forward, time_shifting):
    
    # loss = nn.SmoothL1Loss(beta=1.0)
    # future = torch.tensor(pred_1forward[:,:(-time_shifting)], dtype = torch.float32)
    # target = torch.tensor(pred[:,time_shifting:], dtype = torch.float32)
    # output = loss(future, target)

    future = np.array(pred_1forward[:,:(-time_shifting)])
    target = np.array(pred[:,time_shifting:])
    output = smooth_l1_loss(future, target)

    #return output.item()
    return output

def calculate_consistency(exos_list, pred_exo_list):
        pred_len = 5
        pred = dict()
        pred_1forward = dict()
        compare, compare_1forward = [], []
        for timestep in exos_list.keys():
            if timestep == 0:
                for agent_index in range(len(exos_list[timestep])):
                    agent_id = exos_list[timestep][agent_index]['id']
                    pre = np.array([pred_exo_list[timestep][j][agent_index]['pos'] for j in range(pred_len)])
                    pred[agent_id] = pre
            else:
                pred = copy.deepcopy(pred_1forward)
            
            pred_1forward = dict()
            if exos_list.get(timestep+2) is not None and pred_exo_list.get(timestep+2) is not None:
                key_1forward = timestep+1
                for i in range(len(exos_list[key_1forward])):
                    id_1forward = exos_list[key_1forward][i]['id']
                    try:
                        pre_1forward = np.array([pred_exo_list[key_1forward][j][i]['pos'] for j in range(pred_len)])
                    except:
                        print(f"Error: {timestep}, {agent_id}, {i} {key_1forward} {id_1forward}, "
                              f"{key_1forward in exos_list.keys()}, "
                              f"{key_1forward in pred_exo_list.keys()}, "
                              f"{len(exos_list)}, {len(pred_exo_list)}")
                        assert False
                    pred_1forward[id_1forward] = pre_1forward
            
            for timestep in pred.keys():
                if timestep in pred_1forward:
                    compare.append(pred[timestep])
                    compare_1forward.append(pred_1forward[timestep])
        
        compare = np.array(compare)
        compare_1forward = np.array(compare_1forward)
        tem_cos = consistency(compare,compare_1forward,1)
        #print("Num of calculas:", compare.shape[0])
        #print("Temporal Consistency for the closest Prediction:", tem_cos)

        return tem_cos
    
