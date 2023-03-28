#!/home/phong/anaconda3/envs/HiVT/bin/python

import Pyro4
import numpy as np  # Your motion prediction service module
import argparse
import logging
import sys
import time
import logging.handlers

import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()

MAX_HISTORY_MOTION_PREDICTION = 20

print(f"Python executable: {sys.executable}")

from moped_implementation.planner_wrapper import PlannerWrapper


@Pyro4.expose
class MotionPredictionService(object):

    def __init__(self):
        self.pred_len = 10 # Predict 10 frames into the future as 10 * 0.3 = 3 seconds so it is good enough
        self.planner = PlannerWrapper(pred_len=self.pred_len)


    def predict(self, agents_data):
        '''
            :param data: a dictionary {'agent_id': int, 'agent_history': [(x1,y1), (x2,y2), ...], 'agent_type': int, 'is_ego': bool}
                    Length of agent_history is any arbitrary but must be less than MAX_HISTORY_MOTION_PREDICTION
            
            Return
            :return: a dictionary {'agent_id': int, 'agent_prediction': [(x1,y1), (x2,y2), ...], 'agent_prob': float}
                    Length of agent_prediction is self.pred_len
        '''
        # Step 1. From agents_data, build numpy array (number_agents, observation_len, 2)
        
        agent_id_list = []
        xy_pos_list = []
        ego_id = -1
        for agent_id, agent_data in agents_data.items():
            if agent_data['is_ego'] == False: # -1 is the ego agent, so we add at last
                xy_pos = np.array(agent_data['agent_history'])
                # first axis, we pad width=0 at beginning and pad width=MAX_HISTORY_MOTION_PREDICTION-xy_pos.shape[0] at the end
                # second axis (which is x and y axis), we do not pad anything as it does not make sense to pad anything
                xy_pos = np.pad(xy_pos, pad_width=((0, MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0]), (0, 0)),
                                mode="edge")
                xy_pos_list.append(xy_pos)
                agent_id_list.append(agent_id)
            else:
                ego_id = agent_id

        # Add ego agent at last
        ego_agent_data = agents_data[ego_id] # Ego agent has id -1
        xy_pos = np.array(ego_agent_data['agent_history'])
        xy_pos = np.pad(xy_pos, pad_width=((0, MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0]), (0, 0)),
                                mode="edge")
        xy_pos_list.append(xy_pos)
        agent_id_list.append(ego_id)

        # Creating history
        agents_history = np.stack(xy_pos_list)  # Shape (number_agents, observation_len, 2)

        #print(f"Agents history is {agents_history}")
        #print(f"Shape of agents history is {agents_history.shape}")
            
        try:
            probs, predictions = self.planner.do_predictions(agents_history)
        except Exception as e:
            logging.info(f"Error in prediction: {e} with inputs {agents_history}")
            probs = np.ones(agents_history.shape[0])
            # Predictions is the last known position but with shape (number_agents, self.pred_len, 2)
            predictions = agents_history[:, -self.pred_len:, :]

        #print(f"Predictions are {predictions}")

        # Build response:
        data_response = {}
        for i, agentID in enumerate(agent_id_list):
            prob_info = probs[i]
            agent_predictions = [tuple(row) for row in predictions[i]]

            data_response[agentID] = {'agent_prediction': agent_predictions, 'agent_prob': prob_info, 'agent_id': agentID}

        return data_response

def main(args):
    Pyro4.Daemon.serveSimple(
        {
            MotionPredictionService: "mopedservice.warehouse",
        },
        host=args.host,
        port=args.mopedpyroport,
        ns=False,
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--mopedpyroport',
        metavar='P',
        default=8300,
        type=int,
        help='TCP port to listen to (default: 8300)')
    args = argparser.parse_args()

    max_bytes = 500 * 1024 * 1024  # 500Mb
    backup_count = 1  # keep only the latest file
    filename = f"{current_dir}/logpyro4moped.txt"

    handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    formatter = logging.Formatter(
        '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        handlers=[handler],
        level=logging.DEBUG,
    )

    logging.info(sys.executable)
    logging.info(f"Running PYRO:mopedserve.warehouse on {args.host}:{args.mopedpyroport}")

    main(args)
