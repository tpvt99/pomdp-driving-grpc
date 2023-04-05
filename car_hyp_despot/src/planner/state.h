#ifndef AGENT_STATE_H
#define AGENT_STATE_H
#include <vector>
#include <utility>

#include "coord.h"
#include "param.h"
#include "path.h"
#include "utils.h"
#include "despot/interface/pomdp.h"
#include "moped_param.h"
#include <iomanip>

using namespace std;

using namespace despot;

enum AgentType { car=0, ped=1, num_values=2};

struct CoordHistory {
    std::vector<COORD> coord_history;
    int MAX_HISTORY = MopedParams::MAX_HISTORY_MOTION_PREDICTION;
    int true_history;
    double last_update_time; // time of the last update. We allow Add only if update_time - last_update_time > time_per_move
    CoordHistory() {
        true_history = 0;
        last_update_time = 0;
    }
    std::vector<double> time_history;

    void Add(COORD coord, double update_time, double time_per_move) {
        // I choose 0.85 as the sleeping for in DESPOT sleep only 0.9 of time_per_move
        if ( (coord_history.size() > 0) && (update_time - last_update_time) < 0.85*time_per_move  && (time_per_move > 0)) {
            bool condition1 = (coord_history.size() > 0);
            bool condition2 = ((update_time - last_update_time) < 0.9 * time_per_move);
            bool condition3 = (time_per_move > 0);

            //std::cout << "Condition : " << condition1 << " " << condition2 << " " << condition3 << std::endl;
            return;
        }
        if (coord_history.size() < MAX_HISTORY) {
            coord_history.push_back(coord);
            time_history.push_back(update_time);
        } else {
            coord_history.erase(coord_history.begin());
            time_history.erase(time_history.begin());
            coord_history.push_back(coord);
            time_history.push_back(update_time);
        }
        true_history += 1;
        last_update_time = update_time;
    }

    void CoordText(std::ostream& out) const {

        std::string str = "Hist: " + std::to_string(coord_history.size()) + ", true hist: " + std::to_string(true_history) + " ";

        for (int i = 0; i < coord_history.size(); i++) {
            str += ". [" + std::to_string(i) + "] x:" + std::to_string(coord_history.at(i).x) + " y:" + std::to_string(coord_history.at(i).y) + " t:" + std::to_string(time_history.at(i)) + " ";
        }

        out << str;
    }
        
};

struct AgentStruct {

    AgentStruct(){
        set_default_values();
    }

    AgentStruct(COORD a, int b, int c) {
        set_default_values();
        pos = a;
        intention = b;
        id = c;
    }

    AgentStruct(COORD a, int b, int c, float _speed) {
        set_default_values();
        pos = a;
        intention = b;
        id = c;
        speed = _speed;
    }

    void set_default_values(){
        intention = -1;
        id = -1;
        speed = 0.0;
        mode = 1; //PED_DIS
        type = AgentType::car;
        pos_along_path = 0;
        bb_extent_x = 0;
        bb_extent_y = 0;
        heading_dir = 0;
        cross_dir = 0;
    }

    CoordHistory coordHistory;
    COORD pos;
    int mode;
    int intention; // intended path
    int pos_along_path; // traveled distance along the path
    int cross_dir;
    int id;
    AgentType type;
    double speed;
    COORD vel;
    double heading_dir;
    double bb_extent_x, bb_extent_y;

    void Text(std::ostream& out) const {
        out << "agent: id / pos / speed / vel / intention / dist2car / infront =  "
            << id << " / "
            << "(" << pos.x << ", " << pos.y << ") / "
            << speed << " / "
            << "(" << vel.x << ", " << vel.y << ") / "
            << intention << " / "
            << " (mode) " << mode
            << " (type) " << type
            << " (bb) " << bb_extent_x
            << " " << bb_extent_y
            << " (cross) " << cross_dir
            << " (heading) " << heading_dir << endl;
    }

    void ShortText(std::ostream& out) const {
        out << "Agent: id / type / pos / heading / vel: "
            << id << " / "
            << type << " / "
            << pos.x << "," << pos.y << " / "
            << heading_dir << " / "
            << vel.x << "," << vel.y << endl;
    }

    void PhongAgentText(std::ostream& out) const {
        out << " Agent - id " << std::dec << id << ", cur pos: "
            << std::fixed << std::setprecision(2) << pos.x << ", " << pos.y
            << ", cur yaw/s/v.x/v.y: " << heading_dir << "/"
            << std::setprecision(2) << speed << "/"
            << std::setprecision(2) << vel.x << "/"
            << std::setprecision(2) << vel.y << ".";
        coordHistory.CoordText(out);
        out << endl;
    }
};

struct CarStruct {
    CoordHistory coordHistory;
    COORD pos;
    double vel;
    double heading_dir;/*[0, 2*PI) heading direction with respect to the world X axis */

    void PhongCarText(std::ostream& out) const {
        out << " Car - id " << ", cur pos: "
            << std::fixed << std::setprecision(2) << pos.x << ", " << pos.y
            << ", cur yaw/vel: " << heading_dir << "."
            << std::setprecision(2) << vel << ".";
        coordHistory.CoordText(out);
        out << endl;
    }
};

class PomdpState : public State {
public:
    CarStruct car;
    int num;
    AgentStruct agents[ModelParams::N_PED_IN];

    float time_stamp;

    PomdpState() {time_stamp = -1; num = 0;}

    string Text() const {
        return concat(car.vel);
    }

    void PomdpStateText(std::ostream& out) {
        out << "CarStruct address: " << &car << ". Car Info: ";
        car.PhongCarText(out);
        for (int i = 0; i < num; i++) {
            AgentStruct agent = agents[i];
            agent.PhongAgentText(out);
        }
        out << endl;
    }
};

class PomdpStateWorld : public State {
public:
    CarStruct car;
    int num;
    AgentStruct agents[ModelParams::N_PED_WORLD];

    float time_stamp;

    PomdpStateWorld() {time_stamp = -1; num = 0;}

    string Text() const {
        return concat(car.vel);
    }

    void assign(PomdpStateWorld& src){
        logi << "[PHONG] assign PomdpStateWorld" << endl;
        car.pos = src.car.pos;
        car.vel = src.car.vel;
        car.heading_dir = src.car.heading_dir;
        car.coordHistory = src.car.coordHistory;
        num = src.num;
        for (int i = 0; i < num; i++){
            agents[i].pos = src.agents[i].pos;
            agents[i].coordHistory = src.agents[i].coordHistory;
            agents[i].mode = src.agents[i].mode;
            agents[i].intention = src.agents[i].intention;
            agents[i].pos_along_path = src.agents[i].pos_along_path;
            agents[i].cross_dir = src.agents[i].cross_dir;
            agents[i].id = src.agents[i].id;
            agents[i].type = src.agents[i].type;
            agents[i].speed = src.agents[i].speed;
            agents[i].vel = src.agents[i].vel;
            agents[i].bb_extent_x = src.agents[i].bb_extent_x;
            agents[i].bb_extent_y = src.agents[i].bb_extent_y;
            agents[i].heading_dir = src.agents[i].heading_dir;
        }
        time_stamp = src.time_stamp;
    }

    void PomdpStateWorldText(std::ostream& out) {
        out << "CarStruct address: " << &car << ". Car Info: ";
        car.PhongCarText(out);
        for (int i = 0; i < num; i++) { // +2 is just to make sure that over than nums is not
            AgentStruct agent = agents[i];
            agent.PhongAgentText(out);
        }
        out << endl;
    }
};

#endif
