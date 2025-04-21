import gymnasium as gym
import numpy as np
import traci

class SingleAgentCrosswalkEnv(gym.Env):
    def __init__(self,
                 net_file, route_file,
                 sumo_binary="sumo", use_gui=False,
                 max_steps=1000,
                 alpha: float = 0.05,   # frustration growth rate
                 gamma: float = 1.0):   # frustration weight
        super().__init__()
        self.sumo_binary = sumo_binary
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0

        self.net_file = net_file
        self.route_file = route_file
        self.sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--start"
        ]

        self.n_phases = 7
        self.action_space = gym.spaces.Discrete(self.n_phases)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(12,), dtype=np.float32)

        self.crosswalk_ids = [":TL_c0", ":TL_c1", ":TL_c2", ":TL_c3"]
        self.vehicle_edges = ["N2TL", "E2TL", "S2TL", "W2TL"]

        # frustration hyper-parameters
        self.alpha = alpha
        self.gamma = gamma

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        self.step_count = 0
        traci.start(self.sumo_cmd)
        return self._get_observation(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        # Group 1 green phases (actions 1 or 2) → enforce yellow phase 3
        if action in (1, 2):
            traci.trafficlight.setPhase("TL", action)
            traci.simulationStep()
            self.step_count += 1

            traci.trafficlight.setPhase("TL", 3)
            traci.simulationStep()
            self.step_count += 1

        # Group 2 green phases (actions 5 or 6) → enforce yellow phase 7 (index 6)
        elif action in (5, 6):
            traci.trafficlight.setPhase("TL", action)
            traci.simulationStep()
            self.step_count += 1

            traci.trafficlight.setPhase("TL", 6)
            traci.simulationStep()
            self.step_count += 1

        # All other phases apply normally
        else:
            traci.trafficlight.setPhase("TL", action)
            traci.simulationStep()
            self.step_count += 1

        obs    = self._get_observation()
        reward = self._compute_reward()
        done   = self.step_count >= self.max_steps
        return obs, reward, done, False, {}

    def close(self):
        if traci.isLoaded():
            traci.close()

    def _get_observation(self):
        obs = []
        for cross_id in self.crosswalk_ids:
            ped_ids = traci.edge.getLastStepPersonIDs(cross_id)
            num_waiting = 0
            max_wait = 0
            for pid in ped_ids:
                w = traci.person.getWaitingTime(pid)
                if w > 0:
                    num_waiting += 1
                    max_wait = max(max_wait, w)
            obs.extend([num_waiting, max_wait])
        for edge in self.vehicle_edges:
            obs.append(traci.edge.getLastStepVehicleNumber(edge))
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        # total pedestrian wait
        total_ped_wait = sum(
            traci.person.getWaitingTime(pid)
            for pid in traci.person.getIDList()
        )
        # total vehicle delay
        total_veh_delay = sum(
            traci.edge.getWaitingTime(edge)
            for edge in self.vehicle_edges
        )
        # frustration penalty
        total_frustration = sum(
            np.exp(self.alpha * traci.person.getWaitingTime(pid))
            for pid in traci.person.getIDList()
        )
        # combined reward
        reward = -(
            0.5 * total_ped_wait
            + 0.5 * total_veh_delay
            + self.gamma * total_frustration
        )
        return reward

    def _check_termination(self):
        return self.step_count >= self.max_steps
