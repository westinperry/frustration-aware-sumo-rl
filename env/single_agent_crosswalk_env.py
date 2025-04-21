import gymnasium as gym
import numpy as np
import traci

class SingleAgentCrosswalkEnv(gym.Env):
    def __init__(
                 self,
                 net_file, route_file,
                 sumo_binary="sumo", use_gui=False,
                 max_steps=1000,
                 alpha: float = 0.05,
                 gamma: float = 1.0):
        super().__init__()
        self.sumo_binary = sumo_binary
        self.use_gui     = use_gui
        self.max_steps   = max_steps
        self.step_count  = 0

        self.net_file   = net_file
        self.route_file = route_file
        self.sumo_cmd   = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--start"
        ]

        # Define mapping for agent actions to SUMO phases:
        # - action 0 → phase 0 (traffic green + pedestrian walk at crosswalk 0)
        # - action 1 → phase 4 (traffic green + pedestrian walk at crosswalk 1)
        # - action 2 → phase 3 (all-crosswalk green)
        # - action 3 → phase 7 (all-red clearance)
        self.phase_map = {
            0: 0,
            1: 4,
            2: 3,
            3: 7
        }
        self.action_space = gym.spaces.Discrete(len(self.phase_map))

        # Observation: 4 crosswalks (num_wait, max_wait each) + 4 vehicle counts = 12 dims
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(12,), dtype=np.float32)

        self.crosswalk_ids = [":TL_c0", ":TL_c1", ":TL_c2", ":TL_c3"]
        self.vehicle_edges = ["N2TL", "E2TL", "S2TL", "W2TL"]

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
        # Handle numpy array or list inputs
        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(np.array(action).item())
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        # Map to actual SUMO phase index and set it
        phase_id = self.phase_map[action]
        traci.trafficlight.setPhase("TL", phase_id)

        # Wait out the full phase duration before returning control
        duration = traci.trafficlight.getPhaseDuration("TL")
        for _ in range(int(duration)):
            traci.simulationStep()
            self.step_count += 1

        # Get next observation and compute reward
        obs    = self._get_observation()
        reward = self._compute_reward()
        done   = self.step_count >= self.max_steps
        return obs, reward, done, False, {}

    def close(self):
        if traci.isLoaded():
            traci.close()

    def _get_observation(self):
        obs = []
        for cid in self.crosswalk_ids:
            peds = traci.edge.getLastStepPersonIDs(cid)
            num_wait, max_wait = 0, 0
            for pid in peds:
                w = traci.person.getWaitingTime(pid)
                if w > 0:
                    num_wait += 1
                    max_wait = max(max_wait, w)
            obs.extend([num_wait, max_wait])
        for edge in self.vehicle_edges:
            obs.append(traci.edge.getLastStepVehicleNumber(edge))
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        total_ped_wait = sum(traci.person.getWaitingTime(pid)
                             for pid in traci.person.getIDList())
        total_veh_delay = sum(traci.edge.getWaitingTime(edge)
                              for edge in self.vehicle_edges)
        total_frustration = sum(np.exp(self.alpha * traci.person.getWaitingTime(pid))
                                for pid in traci.person.getIDList())
        return -(0.5 * total_ped_wait + 0.5 * total_veh_delay + self.gamma * total_frustration)

    def _check_termination(self):
        return self.step_count >= self.max_steps
