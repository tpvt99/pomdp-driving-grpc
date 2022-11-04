#!/home/cunjun/anaconda3/envs/conda38/bin/python

# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
import time
import sys

import grpc
import agentinfo_pb2
import agentinfo_pb2_grpc
import numpy as np
from moped_implementation.planner_wrapper import PlannerWrapper



MAX_HISTORY_MOTION_PREDICTION = 20

class Greeter(agentinfo_pb2_grpc.MotionPredictionServicer):

    def Predict(self, request, context):
        xy_pos_list = []
        agent_id_list = []  # To read back when return results

        start = time.time()
        for agentInfo in request.agentInfo:
            x_pos = np.array(agentInfo.x)
            y_pos = np.array(agentInfo.y)
            #logging.info(f"Shape of x_pos is {x_pos.shape} and y_pos is {y_pos.shape}")
            xy_pos = np.concatenate([x_pos[..., np.newaxis], y_pos[..., np.newaxis]], axis=1)  # shape (n,2)
            # first axis, we pad width=0 at beginning and pad width=MAX_HISTORY_MOTION_PREDICTION-xy_pos.shape[0] at the end
            # second axis (which is x and y axis), we do not pad anything as it does not make sense to pad anything
            xy_pos = np.pad(xy_pos, pad_width=((0, MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0]), (0, 0)),
                            mode="edge")
            xy_pos_list.append(xy_pos)
            #logging.info(f"[P] Shape of x_pos is {x_pos.shape} and y_pos is {y_pos.shape} and after padd {xy_pos.shape}")

            agent_id_list.append(agentInfo.agentId)

        logging.info(f"Time for building agentInfo array: {time.time() - start}")
        prediction_time = time.time()

        # Shape (number_agents, observation_len, 2)
        agents_history = np.stack(xy_pos_list)
        planner = PlannerWrapper()
        #logging.info(f"[P] Shape of agents_history is {agents_history.shape} and list length {len(agent_id_list)}")

        #logging.info(f"Time for preprocessing: {time.time() - start}")

        # Simple simulation
        probs, predictions = planner.constant_velocity(agents_history) # probs shape (number_agents,) predictions shape (number_agents, pred_len, 2)
        #probs, predictions = planner.constant_acceleration(agents_history)
        #probs, predictions = planner.knn_map_nosocial(agents_history)

        #logging.info(f"[P] Shape of probs is {probs.shape} and predictions is {predictions.shape}")

        logging.info(f"Time for prediction: {time.time() - prediction_time}")
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
        logging.info(f"Time for producing response: {end - response_time}")
        logging.info(f"Time for running end-to-end: {end - start}")

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1000))
    agentinfo_pb2_grpc.add_MotionPredictionServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(filename="/home/cunjun/p3_catkin_ws_new/src/moped/logfilexx.txt",
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.info(sys.executable)
    serve()
