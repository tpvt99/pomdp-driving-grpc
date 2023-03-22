#!/home/cunjun/anaconda3/envs/conda38/bin/python


import Pyro4
import numpy as np  # Your motion prediction service module
import argparse
import logging
import sys
import time

import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()

MAX_HISTORY_MOTION_PREDICTION = 20

from moped_implementation.planner_wrapper import PlannerWrapper


@Pyro4.expose
class MotionPredictionService(object):

    def __init__(self):
        self.planner = PlannerWrapper()


    def predict(self, agents_data):
        '''
            :param data: a dictionary {'agent_id': int, 'agent_history': [(x1,y1), (x2,y2), ...], 'agent_type': int}
        '''
        # Step 1. From agents_data, build numpy array (number_agents, observation_len, 2)
        start_t = time.time()
        xy_pos_list = []
        for agent_id, agent_data in agents_data.item():
            xy_pos = np.array(agent_data['agent_history'])
            # first axis, we pad width=0 at beginning and pad width=MAX_HISTORY_MOTION_PREDICTION-xy_pos.shape[0] at the end
            # second axis (which is x and y axis), we do not pad anything as it does not make sense to pad anything
            xy_pos = np.pad(xy_pos, pad_width=((0, MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0]), (0, 0)),
                            mode="edge")
            xy_pos_list.append(xy_pos)


        agents_history = np.stack(xy_pos_list)  # Shape (number_agents, observation_len, 2)

            
        try:
            probs, predictions = self.planner.do_predictions(agents_history)
        except Exception as e:
            logging.info(f"Error in prediction: {e} with inputs {agents_history}")
            probs = np.ones(agents_history.shape[0])
            # Predictions is the last known position but with shape (number_agents, 1, 2)
            predictions = agents_history[:, [-1], :]


        response_time = time.time()

        # Build response:
        response = agentinfo_pb2.PredictionResponse()
        for i, id in enumerate(agent_id_list):
            prob_info = agentinfo_pb2.ProbabilityInfo(prob = probs[i], agentId = id)
            agent_info = agentinfo_pb2.AgentInfo()
            agent_info.agentId = id
            agent_info.x.append(predictions[i][0][0]) # Get number_agent_id of 1st axis, of first pred of 2nd axis, of x of 3rd axis
            agent_info.y.append(predictions[i][0][1]) # Get number_agent_id of 1st axis, of first pred of 2nd axis, of y of 3rd axis

            response.agentInfo.append(agent_info)
            response.probInfo.append(prob_info)

        end = time.time()
        #logging.info(f"Time for producing response: {end - response_time}")
        #logging.info(f"Time for running end-to-end: {end - start}")

        return response

def main(args):
    Pyro4.Daemon.serveSimple(
        {
            MotionPredictionService: "motion_prediction_service",
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
        default=8200,
        type=int,
        help='TCP port to listen to (default: 8200)')
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

    main(args)
