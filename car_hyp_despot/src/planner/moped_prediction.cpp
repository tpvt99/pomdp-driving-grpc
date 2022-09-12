//
// Created by cunjun on 25/8/22.
//

#include "moped_prediction.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

std::string target_str =  "localhost:50051";
MotionPredictionClient mopedClient(
        grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

MotionPredictionClient::MotionPredictionClient(std::shared_ptr<Channel> channel)
            : stub_(agentinfo::MotionPrediction::NewStub(channel)) {}

// Assembles the client's payload, sends it and presents the response back
// from the server.
std::map<int, std::vector<double>> MotionPredictionClient::Predict(std::vector<AgentStruct> neighborAgents) {
        // Data we are sending to the server.
        agentinfo::PredictionRequest request;

        for (const AgentStruct &tempAgent : neighborAgents) {
            std::vector<COORD> hist = tempAgent.coordHistory.coord_history;
            agentinfo::AgentInfo *agentInfo = request.add_agentinfo();
            for (unsigned int i = 0; i < hist.size(); i++) {
                agentInfo->add_x(hist[i].x);
                agentInfo->add_y(hist[i].y);

            }
            agentInfo->set_agentid(tempAgent.id);
            agentInfo->set_agenttype(tempAgent.type);
        }

        // Container for the data we expect from the server.
        agentinfo::PredictionResponse reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->Predict(&context, request, &reply);

        // Act upon its status.
        std::map<int, std::vector<double>> results;

        if (status.ok()) {
            std::cout << "Success" << std::endl;
            for (int i = 0; i < reply.agentinfo_size(); i++) {
                std::vector<double> returnAgentInfo;

                agentinfo::ProbabilityInfo *probInfo = reply.mutable_probinfo(i);
                agentinfo::AgentInfo *agentInfo = reply.mutable_agentinfo(i);

                //std::cout << "AgentInfo: id: " << agentInfo->agentid() << " x: " << agentInfo->x(0) << std::endl;
                //std::cout << "Prob Info: id: " << probInfo->agentid() << " prob: " << probInfo->prob() << std::endl;
                returnAgentInfo.push_back(probInfo->agentid());
                returnAgentInfo.push_back(agentInfo->x(0));
                returnAgentInfo.push_back(agentInfo->y(0));

                results.insert({agentInfo->agentid(), returnAgentInfo});
            }
            return results;
        } else {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
            std::cout << "RPC Failed!" << std::endl;
            return results;
        }
}

std::map<int, std::vector<double>> callPython(std::vector<AgentStruct> neighborAgents) {
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint specified by
    // the argument "--target=" which is the only expected argument.
    // We indicate that the channel isn't authenticated (use of
    // InsecureChannelCredentials()).
    return mopedClient.Predict(neighborAgents);
}
