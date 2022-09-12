#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <numeric>

#include <core/globals.h>
#include <solver/despot.h>
#include <despot/util/logging.h>

#include "config.h"
#include "coord.h"
#include "param.h"
#include "crowd_belief.h"
#include "context_pomdp.h"
#include "world_model.h"
#include "simulator_base.h"
#include "moped_prediction.hpp"

int use_att_mode = 2;		// onlu use att mode
vector<State*> particles_;

HiddenStateBelief::HiddenStateBelief(int num_intentions, int num_modes) {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::HiddenStateBelief (int intentions, int mode) Constructor" << endl;
    Resize(num_intentions, num_modes);
}

void HiddenStateBelief::Reset() {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::Reset Reset" << endl;
    for (std::vector<double>& intention_probs: probs_) {
        std::fill(intention_probs.begin(), intention_probs.end(),
                  1.0/intention_probs.size()/probs_.size());
    }
}

void HiddenStateBelief::Resize(int new_intentions) {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::Resize (int intentions)" << endl;
    int mode_size = probs_.size();
    Resize(new_intentions, mode_size);
}

void HiddenStateBelief::Resize(int new_intentions, int new_modes) {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::Resize (int intentions, int mode)" << endl;
    probs_.resize(new_modes);
    for (std::vector<double>& intention_probs: probs_) {
        intention_probs.resize(new_intentions);
        std::fill(intention_probs.begin(), intention_probs.end(), 1.0/new_intentions/new_modes);
    }
}

double gaussian_prob(double x, double stddev) {
    double a = 1.0 / stddev / sqrt(2 * M_PI);
    double b = -x * x / 2.0 / (stddev * stddev);
    return a * exp(b);
}

double TransitionLikelihood(const COORD& past_pos, const COORD& cur_pos, const COORD& pred_pos) {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] TransitionLikelihood" << endl;

    const double K = 0.001;

    double goal_dist = (pred_pos - past_pos).Length();
    double move_dist = (cur_pos - past_pos).Length();

    double angle_error = COORD::Angle(past_pos, cur_pos, pred_pos, 0.0);
    double angle_prob = gaussian_prob(angle_error,
                                      ModelParams::NOISE_GOAL_ANGLE) + K;

    double dist_error = move_dist - goal_dist;
    double dist_prob = gaussian_prob(dist_error,
                                     ModelParams::NOISE_PED_VEL / ModelParams::CONTROL_FREQ) + K;

    if (isnan(angle_prob) || isnan(dist_prob))
    ERR("Get transition likelihood as NAN");

    return angle_prob * dist_prob;
}

void HiddenStateBelief::Update(WorldModel& model, AgentStruct& past_agent, const AgentStruct& cur_agent,
                               int intention_id, int mode_id, std::map<int, std::vector<double>> prediction_result) {

    AgentStruct predicted_agent = past_agent;

    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] HiddenStateBelief::Update 1 Step before" << endl;
//        logi << "HSB::Update before pred_agent: id: " << &predicted_agent << " ";
//        predicted_agent.PhongAgentText(cout);
//        logi << "HSB::Update before past_agent: " << &past_agent << " ";
//        past_agent.PhongAgentText(cout);
//        logi << "HSB::Update before cur_agent: "<< &cur_agent << " ";
//        const_cast<AgentStruct &>(cur_agent).PhongAgentText(cout);
    }

    if (MopedParams::USE_MOPED) {
        model.PhongAgentStep(predicted_agent, intention_id, prediction_result);
    } else {
        if (past_agent.type == AGENT_ATT)
            model.GammaAgentStep(predicted_agent, intention_id);
        else
            model.AgentStepPath(predicted_agent);
    }

    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::Update 2 Finding likelihood" << endl;

    double likelihood = TransitionLikelihood(past_agent.pos, cur_agent.pos, predicted_agent.pos);

    probs_[mode_id][intention_id] = likelihood * probs_[mode_id][intention_id];

    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] HiddenStateBelief::Update 1 Step after" << endl;
//        logi << "HSB::Update after pred_agent: ";
//        predicted_agent.PhongAgentText(cout);
//        logi << "HSB::Update after past_agent: ";
//        past_agent.PhongAgentText(cout);
//        logi << "HSB::Update after cur_agent: ";
//        const_cast<AgentStruct &>(cur_agent).PhongAgentText(cout);
    }
}

void HiddenStateBelief::Normalize() {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::Normalize" << endl;

    double total_prob = 0;
    for (auto& intention_probs: probs_) {
        total_prob += std::accumulate(intention_probs.begin(), intention_probs.end(), 0.0);
    }

    if(total_prob == 0)
    ERR("total_prob == 0");

    for (auto& intention_probs: probs_) {
        std::transform(intention_probs.begin(), intention_probs.end(), intention_probs.begin(),
                       std::bind(std::multiplies<double>(), std::placeholders::_1, 1.0/total_prob));
    }
}

void HiddenStateBelief::Sample(int& intention_id, int& mode_id) {
    this->Normalize();
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] HiddenStateBelief::Sample" << endl;

    double r = Random::RANDOM.NextDouble();

    if (use_att_mode == 0) {
        for (int mode = 0; mode < probs_.size(); mode++) {
            auto& goal_probs = probs_[mode];
            for (int intention = 0; intention < goal_probs.size(); intention++) {
                r -= probs_[mode][intention];
                if (r <= 0.001) {
                    intention_id = intention;
                    mode_id = mode;
                    break;
                }
            }
            if (r <= 0.001)
                break;
        }
    } else {
        int mode = (use_att_mode <= 1) ? AGENT_DIS : AGENT_ATT;
        auto& goal_probs = probs_[mode];
        double total_prob = std::accumulate(goal_probs.begin(), goal_probs.end(), 0.0);
        for (int intention = 0; intention < goal_probs.size(); intention++) {
            r -= probs_[mode][intention] / total_prob;
            if (r <= 0.001) {
                intention_id = intention;
                mode_id = mode;
                break;
            }
        }
    }

    if (r > 0.001)
    ERR("Sampling belief failed");
}

void AgentBelief::Reset(int new_intentions) {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] AgentBelief::Reset:" << endl;

    if (new_intentions != belief_->size(1))
        belief_->Resize(new_intentions);
    else
        belief_->Reset();
}

void AgentBelief::Sample(int& intention, int& mode) const {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] AgentBelief::Sample:" << endl;

    belief_->Sample(intention, mode);
}

void AgentBelief::Update(WorldModel& model, const AgentStruct& cur_agent, int num_intentions,
                         std::map<int, std::vector<double>> prediction_result) {
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] AgentBelief::Update:" << endl;

    if (num_intentions != belief_->size(1))
        belief_->Resize(num_intentions);



    AgentStruct past_agent = observable_;
    for (int mode = 0; mode < belief_->size(0); mode++)
        for (int intention = 0; intention < belief_->size(1); intention++) {
            belief_->Update(model, past_agent, cur_agent, intention, mode, prediction_result);
        }
    observable_ = cur_agent;
    time_stamp = Globals::ElapsedTime();
}

CrowdBelief::CrowdBelief(const DSPOMDP* model): Belief(model),
                                                world_model_(SimulatorBase::world_model){
}

CrowdBelief::CrowdBelief(const DSPOMDP* model, History history,
                         std::map<double, AgentBelief*> sorted_belief_) : Belief(model),
                                                                          sorted_belief_(sorted_belief_),	world_model_(SimulatorBase::world_model) {
}

CrowdBelief::~CrowdBelief() {
    for (std::map<double,AgentBelief*>::iterator it=sorted_belief_.begin();
         it!=sorted_belief_.end(); ++it) {
        AgentBelief* b = it->second;
        delete b;
    }
}

Belief* CrowdBelief::MakeCopy() const {
    return new CrowdBelief(model_, history_, sorted_belief_);
}


std::vector<State*> CrowdBelief::Sample(int num) const {
    logi << "Sample particles from belief" << endl;
    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Sample:" << endl;
    }

    const ContextPomdp* context_pomdp = static_cast<const ContextPomdp*>(model_);

    if (DESPOT::Debug_mode)
        std::srand(0);

    for (auto* particle: particles_) {
        if (particle->IsAllocated())
            model_->Free(particle);
    }
    particles_.resize(num);

    for (int i = 0; i < num; i++) {
        particles_[i] = model_->Allocate(i, 1.0/num);
    }

    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Sample: Size of particles_: " << particles_.size() << endl;
    }


    for(auto* particle: particles_) {
        PomdpState* state = static_cast<PomdpState*>(particle);
        state->car = car_;
        state->num = min<int>(ModelParams::N_PED_IN, sorted_belief_.size());

        cout << "state->num=" << state->num <<
             ", sorted_belief_.size()=" << sorted_belief_.size() << endl;

        state->time_stamp = Globals::ElapsedTime();

        int agent_id = 0;
        for (std::map<double,AgentBelief*>::const_iterator it=sorted_belief_.begin(); it!=sorted_belief_.end(); ++it) {
            if (agent_id < state->num) {
                if (MopedParams::PHONG_DEBUG) {
                    logi << "[PHONG] [CrowdBelief::Sample] agent before: " << state->agents[agent_id].id <<
                         " intentions: " << state->agents[agent_id].intention << "mode: "
                         << state->agents[agent_id].mode << endl;
                }
                AgentBelief* b = it->second;
                state->agents[agent_id] = b->observable_;
                b->Sample(state->agents[agent_id].intention, state->agents[agent_id].mode);

                logd << "[Sample] agent " << state->agents[agent_id].id <<
                     " num_intentions=" <<
                     world_model_.GetNumIntentions(state->agents[agent_id].id) << endl;

                if (MopedParams::PHONG_DEBUG) {

                    logi << "[PHONG] [CrowdBelief::Sample] agent after: " << state->agents[agent_id].id <<
                         " new intention: " << state->agents[agent_id].intention << "new mode: "
                         << state->agents[agent_id].mode << endl;
                }
                world_model_.ValidateIntention(state->agents[agent_id].id, state->agents[agent_id].intention,
                                               __FUNCTION__, __LINE__);

                agent_id++;
            } else
                break;
        }
    }

    if (abs(State::Weight(particles_) - 1.0) > 1e-6) {
        cerr << "[CrowdBelief::CrowdBelief] Particle weights sum to "
             << State::Weight(particles_) << " instead of 1" << endl;
        ERR("particle sampling error");
    }

    random_shuffle(particles_.begin(), particles_.end());

    return particles_;
}

void CrowdBelief::Update(ACT_TYPE action, OBS_TYPE obs) {
    ERR("this update function is deprecated");
}

void CrowdBelief::Update(ACT_TYPE action, const State* state) {
    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Update" << endl;
    }

    const PomdpStateWorld* observed = static_cast<const PomdpStateWorld*>(state);

    logd << "[CrowdBelief::Update] " << "observed->num=" << observed->num << endl;
    std::map<int, const AgentStruct*> src_agent_map;

    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Update 2.a length of observed->num : " << observed->num << endl;
    }

    for (int i=0;i < observed->num; i++) {
        src_agent_map[observed->agents[i].id] = &(observed->agents[i]);
    }

    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Update 2.b - Done Assigning agents[i] to src_agent_map" << endl;
    }

    // new container for belief update
    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Update 3.a indexed_belief from sorted_belief_: ";
    }

    std::map<int,AgentBelief*> indexed_belief;
    for (std::map<double,AgentBelief*>::iterator it=sorted_belief_.begin(); it!=sorted_belief_.end(); ++it) {
        AgentBelief* agent_belief = it->second;
        indexed_belief[agent_belief->observable_.id] = agent_belief;
        //logi << agent_belief->observable_.id << " ";

    }
    //logi << endl;
    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Update 3.b Done sorted_belief_: " << endl;
    }

    logd << "[CrowdBelief::Update] " << "indexed_belief.size()=" << indexed_belief.size() << endl;

    // update belief
    int n = 0;
    car_ = observed->car;
    if (MopedParams::PHONG_DEBUG) {
        logi << "[PHONG] CrowdBelief::Update 4.a PrepareAttentiveAgentMeanDirs: " << endl;
    }

    world_model_.PrepareAttentiveAgentMeanDirs(state);

    // Build neighbor agents
    std::vector<AgentStruct> neighborAgents;
    for (auto it= src_agent_map.begin(); it!=src_agent_map.end(); ++it) {
        AgentStruct agent = *(it->second);
        neighborAgents.push_back(agent);
    }

    std::map<int, std::vector<double>> predictionResults = callPython(neighborAgents);


    for (auto it= src_agent_map.begin(); it!=src_agent_map.end(); ++it) {
        int id = it->first;
        const AgentStruct& agent = *(it->second);
        auto it1 = indexed_belief.find(id);
        int num_intentions = world_model_.GetNumIntentions(agent.id);
        logd << "[Update] agent " << agent.id << " num_intentions=" << num_intentions << endl;

        if (MopedParams::PHONG_DEBUG)
            logi << "[PHONG] CrowdBelief::Update [Update] 4.b agent " << agent.id << " num_intentions=" << num_intentions << endl;

        if (it1 != indexed_belief.end()) { // existing agents
            AgentBelief* agent_belief = it1->second;
            if (world_model_.NeedBeliefReset(id))
                agent_belief->Reset(num_intentions);
            if (MopedParams::PHONG_DEBUG)
                logi << "[PHONG] CrowdBelief::Update Update 4.c Agent_belief " << agent.id << endl;

            agent_belief->Update(world_model_, agent, num_intentions, predictionResults);
        } else { // new agent
            indexed_belief[id] = new AgentBelief(num_intentions, PED_MODES::NUM_AGENT_TYPES);
            indexed_belief[id]->observable_ = agent;

            if (MopedParams::PHONG_DEBUG)
                logi << "[PHONG] CrowdBelief::Update New 4.d Agent_belief " << agent.id << endl;
        }
    }

    // Py_XDECREF(object); This makes error but I do not know why

    logd << "[CrowdBelief::Update] " << "indexed_belief.size()=" << indexed_belief.size() << endl;

    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] [CrowdBelief::Update] 5 " << "indexed_belief.size()=" << indexed_belief.size() << endl;

    // remove out-dated agents
    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] [CrowdBelief::Update] 6 " << "Removing out-dated agents from indexed_belief: ";

    double time_stamp = Globals::ElapsedTime();
    std::map<int,AgentBelief*>::iterator it = indexed_belief.begin();
    while (it != indexed_belief.end()){
        AgentBelief* agent_belief = it->second;
        if (agent_belief->OutDated(time_stamp)) {
            logd << "[CrowdBelief::Update] " << "agent disappear" << endl;
            logi << agent_belief->observable_.id << " ";
            delete agent_belief;
            it = indexed_belief.erase(it);
        } else
            ++it;
    }
    logi << endl;

    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] [CrowdBelief::Update] 7 " << "Reset agents disappeared less than 2 seconds from indexed_belief: ";

    // agents disappeared less than 2 seconds
    for (auto it=indexed_belief.begin(); it!=indexed_belief.end(); ++it) {
        AgentBelief* agent_belief = it->second;
        if (world_model_.NumPaths(agent_belief->observable_.id) == 0) {
            logd << "[CrowdBelief::Update] " << "cur time_stamp = " << time_stamp
                 << ", belief time_stamp="<< agent_belief->time_stamp << endl;
            agent_belief->Reset(world_model_.GetNumIntentions(agent_belief->observable_.id));
            logi << agent_belief->observable_.id << " ";
        }
    }
    logi << endl;

    logd << "[CrowdBelief::Update] " << "indexed_belief.size()=" << indexed_belief.size() << endl;

    if (MopedParams::PHONG_DEBUG)
        logi << "[PHONG] [CrowdBelief::Update] Regenerated sorted_belief_ from indexed_belief" << "indexed_belief.size()=" << indexed_belief.size() << endl;

    // regenerate ordered belief
    sorted_belief_.clear();
    for (std::map<int,AgentBelief*>::iterator it=indexed_belief.begin(); it!=indexed_belief.end(); ++it) {
        int id = it->first;
        AgentBelief* agent_belief = it->second;
        double dist_to_car = COORD::EuclideanDistance(agent_belief->observable_.pos, car_.pos);
        sorted_belief_[dist_to_car] = agent_belief;
    }

    logd << "[CrowdBelief::Update] " << "sorted_belief_.size()=" << sorted_belief_.size() << endl;
    logi << "[CrowdBelief::Update] " << "sorted_belief_.size()=" << sorted_belief_.size() << endl;

}
