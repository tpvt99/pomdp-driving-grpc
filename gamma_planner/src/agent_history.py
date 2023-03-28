import time


class AgentType:
    car = 0 # Car
    ped = 1 # Pedestrian
    num_values = 2 # Other

class History():
    def __init__(self, max_observations = 20, time_interval = 0.3, time_threshold = 3.0):
        '''
        max_observations: maximum number of observations to store
        time_interval: time interval between two observations
        time_threshold: time threshold to determine if the agent is stuck
        '''
        self.max_observations = max_observations
        self.time_interval = time_interval
        self.time_threshold = time_threshold
        self.exo_history = {}
        self.ego_history = []
        self.last_exo_update = {}
        self.last_ego_update = None
        self.ego_id = -1

    def add_exo_observation(self, agent_id, x, y):
        '''
        Add an exo observation
        '''
        current_time = time.time()
        if agent_id not in self.last_exo_update or (current_time - self.last_exo_update[agent_id]) >= self.time_interval:
            if (current_time - self.last_exo_update.get(agent_id, 0)) > self.time_threshold:
                self.exo_history[agent_id] = []

            if len(self.exo_history[agent_id]) >= self.max_observations:
                self.exo_history[agent_id].pop(0)
            self.exo_history[agent_id].append((x, y))
            self.last_exo_update[agent_id] = current_time

            
    def add_ego_observation(self, x, y):
        current_time = time.time()
        if self.last_ego_update is None or (current_time - self.last_ego_update) >= self.time_interval:
            if (current_time - (self.last_ego_update or 0)) > self.time_threshold:
                self.ego_history = []

            if len(self.ego_history) >= self.max_observations:
                self.ego_history.pop(0)
            self.ego_history.append((x, y))
            self.last_ego_update = current_time

    def get_exo_history(self, agent_id):
        return self.exo_history.get(agent_id, [])

    def get_ego_history(self):
        return self.ego_history
    
    def build_request(self):
        '''
        Build request for MOPED
        '''
        request = {}
        for agent_id, history in self.exo_history.items():
            request[agent_id] = {'agent_id': agent_id, 'agent_type': AgentType.car, 'agent_history': history, 'is_ego': False}

        request[self.ego_id] = {'agent_id': self.ego_id, 'agent_type': AgentType.car, 'agent_history': self.ego_history, 'is_ego': True}
        
        return request
    
    def set_ego_id(self, ego_id):
        self.ego_id = ego_id

    def __str__(self):
        return "Exo History: {}\n Ego History: {}".format(self.exo_history, self.ego_history)