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


import grpc
import agentinfo_pb2
import agentinfo_pb2_grpc
import numpy as np
from moped_implementation.planner_wrapper import PlannerWrapper

logging.basicConfig(filename="/home/cunjun/p3_catkin_ws_new/src/moped/logfilexx.txt",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

MAX_HISTORY_MOTION_PREDICTION = 30

class Greeter(agentinfo_pb2_grpc.MotionPredictionServicer):

    def Predict(self, request, context):
        xy_pos_list = []
        agent_id_list = []  # To read back when return results
        logging.info(f"Receive request")
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
            agent_id_list.append(agentInfo.agentId)

        agents_history = np.stack(xy_pos_list)
        planner = PlannerWrapper()

        logging.info(f"Time for preprocessing: {time.time() - start}")
        start = time.time()

        # Simple simulation
        probs, predictions = planner.constant_velocity(agents_history)

        logging.info(f"Time for prediction: {time.time() - start}")
        start = time.time()

        # Build response:
        response = agentinfo_pb2.PredictionResponse()
        for i, id in enumerate(agent_id_list):
            prob_info = agentinfo_pb2.ProbabilityInfo(prob = probs[i], agentId = id)
            agent_info = agentinfo_pb2.AgentInfo()
            agent_info.agentId = id
            agent_info.x.append(predictions[i][0][0])
            agent_info.y.append(predictions[i][0][1])

            response.agentInfo.append(agent_info)
            response.probInfo.append(prob_info)

        logging.info(f"Time for producing response: {time.time() - start}")
        start = time.time()

        return response



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agentinfo_pb2_grpc.add_MotionPredictionServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
