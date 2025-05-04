import gymnasium as gym
import numpy as np
import traci

class SingleAgentCrosswalkEnv(gym.Env):
    def __init__(self, net_file, route_file,
             sumo_binary="sumo", use_gui=True,
             max_steps=1000, alpha=0.01, gamma=0.0,
             ped_weight=1.0, veh_weight=1.0):
        super().__init__()
        self.sumo_binary = sumo_binary
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0
        self.traci = traci
        self.net_file = net_file
        self.route_file = route_file
        self.sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--start", "false",  # Added comma here
            "--error-log", "sumo_crash.log",
            "--message-log", "sumo_messages.log",
            "--time-to-teleport", "10000",
            "--no-warnings", "true",
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

        self.crosswalk_ids = [":TL_w0", ":TL_w1", ":TL_w2", ":TL_w3"]
        self.vehicle_edges = ["N2TL", "E2TL", "S2TL", "W2TL"]

        self.alpha = alpha
        self.gamma = gamma
        self.ped_weight = ped_weight
        self.veh_weight = veh_weight
        self.last_agent_phase = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        self.step_count = 0
        self.last_agent_phase = None
        self.total_episode_reward = 0.0  # ← Add this line
        traci.start(self.sumo_cmd)

        if self.use_gui:
            for phase in self.agent_action_map.values():
                traci.trafficlight.setPhase("TL", phase)
                for _ in range(5):
                    traci.simulationStep()

        obs = self._get_observation()
        # print(f"[Observation @ reset ] {obs}")
        return obs, {}

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
        self.total_episode_reward += reward  # ← Accumulate episode reward
        done = self._check_termination()

        if done:
            print(f"\n🏁 [Episode Complete] Total Reward: {self.total_episode_reward:.1f}\n")

        # print(f"[Observation @ step {self.step_count:4d}] {obs}")
        return obs, reward, done, False, {}



    def _set_phase_and_step(self, phase_id):
        traci.trafficlight.setPhase("TL", phase_id)
        duration = traci.trafficlight.getPhaseDuration("TL")
        for _ in range(int(duration)):
            traci.simulationStep()
            self.step_count += 1

    def _get_observation(self):
            obs = []

            # 1) count & max‐wait on each pedestrian queue edge
            for eid in self.crosswalk_ids:
                pids = traci.edge.getLastStepPersonIDs(eid)
                num_wait, max_wait = 0, 0
                for pid in pids:
                    w = traci.person.getWaitingTime(pid)
                    if w > 0:
                        num_wait  += 1
                        max_wait  = max(max_wait, w)
                obs.extend([num_wait, max_wait])

            # 2) vehicle counts on each incoming edge
            for edge in self.vehicle_edges:
                obs.append(traci.edge.getLastStepVehicleNumber(edge))

            # 3) debug print to confirm
            # print(f"[Obs] ped_counts&max = {obs[:8]}, veh_counts = {obs[8:]}")

            return np.array(obs, dtype=np.float32)


    def _compute_reward(self):
        # 1) Compute total pedestrian waiting time
        total_ped_wait = sum(
            traci.person.getWaitingTime(pid)
            for pid in traci.person.getIDList()
        )

        # 2) Compute total vehicle delay
        total_veh_delay = sum(
            traci.edge.getWaitingTime(edge)
            for edge in self.vehicle_edges
        )

        # 3) Compute total frustration for pedestrians waiting over 60 seconds
        FRUSTRATION_LIMIT = 10000  # max frustration contribution per pedestrian
        total_frustration = sum(
            min(np.exp(self.alpha * (traci.person.getWaitingTime(pid) - 60)), FRUSTRATION_LIMIT)
            for pid in traci.person.getIDList()
            if traci.person.getWaitingTime(pid) > 60
        )

        # 4) Compute each weighted component
        ped_term = self.ped_weight * total_ped_wait
        veh_term = self.veh_weight * total_veh_delay
        frust_term = self.gamma * total_frustration

        # 5) Sum them (and negate for minimization)
        reward = -(ped_term + veh_term + frust_term)

        # 6) Debug print breakdown
        # print(
        #     f"[Reward Breakdown]\n"
        #     f"  ped_term   = {self.ped_weight} * {total_ped_wait:.1f} = {ped_term:.1f}\n"
        #     f"  veh_term   = {self.veh_weight} * {total_veh_delay:.1f} = {veh_term:.1f}\n"
        #     f"  frust_term = {self.gamma} * {total_frustration:.1f} = {frust_term:.1f}\n"
        #     f"→ reward    = -({ped_term:.1f} + {veh_term:.1f} + {frust_term:.1f}) = {reward:.1f}"
        # )

        return reward



    def _check_termination(self):
        return self.step_count >= self.max_steps

    def close(self):
        if traci.isLoaded():
            traci.close()
