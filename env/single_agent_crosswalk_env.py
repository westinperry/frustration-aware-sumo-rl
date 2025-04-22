import gymnasium as gym
import numpy as np
import traci

class SingleAgentCrosswalkEnv(gym.Env):
    def __init__(self, net_file, route_file,
                 sumo_binary="sumo", use_gui=False,
                 max_steps=1000, alpha=0.05, gamma=0.5):
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
            "--start",
            "--error-log", "sumo_crash.log",
            "--message-log", "sumo_messages.log",
            "--time-to-teleport", "10000",
            "--no-step-log"
        ]

        self.agent_action_map = {
            0: 0,
            1: 3,
            2: 4,
            3: 7
        }

        self.transition_after = {
            0: [1, 2],
            2: [5, 6],
        }

        self.action_space = gym.spaces.Discrete(len(self.agent_action_map))
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(12,), dtype=np.float32)

        self.crosswalk_ids = [":TL_c0", ":TL_c1", ":TL_c2", ":TL_c3"]
        self.vehicle_edges = ["N2TL", "E2TL", "S2TL", "W2TL"]

        self.alpha = alpha
        self.gamma = gamma
        self.last_agent_phase = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        self.step_count = 0
        self.last_agent_phase = None
        traci.start(self.sumo_cmd)
        return self._get_observation(), {}

    def step(self, action):
        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(np.array(action).item())
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        mapped_phase = self.agent_action_map[action]

        if self.last_agent_phase in self.transition_after:
            for phase in self.transition_after[self.last_agent_phase]:
                self._set_phase_and_step(phase)

        self._set_phase_and_step(mapped_phase)
        self.last_agent_phase = action

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        return obs, reward, done, False, {}

    def _set_phase_and_step(self, phase_id):
        traci.trafficlight.setPhase("TL", phase_id)
        duration = traci.trafficlight.getPhaseDuration("TL")
        for _ in range(int(duration)):
            traci.simulationStep()
            self.step_count += 1

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

    def close(self):
        if traci.isLoaded():
            traci.close()
