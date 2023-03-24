#!/home/phong/anaconda3/envs/HiVT/bin/python

import Pyro4

def create_sample_data():

    # Build request
    raw_data = {10: {"x": [150,152,160], "y":[250.2, 252.2, 255.2], "type": 1},
            20: {"x": [151,159,165, 170, 180], "y":[255.2, 259.2, 260.2, 240, 245], "type": 1},
            3: {"x": [160.1, 166], "y": [256, 257], "type": 2}}
        
    sample_data = {}
    for agent_id, agent_info in raw_data.items():
        agent_type = agent_info["type"]  # Change this if needed
        observations = [(x, y) for x, y in zip(agent_info["x"], agent_info["y"])]
        sample_data[agent_id] = {'agent_id': agent_id, 'agent_type': agent_type, 'agent_history': observations}
    return sample_data

def main():
    agent_data = create_sample_data()  # Modify the number of agents if needed
    agent_predictor = Pyro4.Proxy("PYRO:mopedservice.warehouse@localhost:8300")

    print(f"Data is {agent_data}")

    future_frames = agent_predictor.predict(agent_data)

    for agent_id, agent_data in future_frames.items():
        print(f"Agent ID {agent_id}: {agent_data['agent_id']}  Agent prob: {agent_data['agent_prob']} Agent data: {agent_data['agent_prediction']}")

if __name__ == "__main__":
    main()
