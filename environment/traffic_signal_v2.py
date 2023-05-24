"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gym import spaces
import math

from networkdata import NetworkData
nd = NetworkData('net_jkt-new/jkt-new.net.xml')
netdata = nd.get_net_data()


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 3 + 2.5 # 3(vehSize) + 2.5(minGap)

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo

        self.density_in_maxflow = 1#veh/km
        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}
        self.out_lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.out_lanes} # out coming lanes
        self.mp_lanes = self.max_pressure_lanes()

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        #self.last_reward = self._waiting_time_reward()
        self.last_reward = self._pressure_reward()
        #self.last_reward = -(0.5*self._weight_reward() + 0.5*self._new_queue_reward())
        #self.last_reward = self.exponential_reward()
        return self.last_reward

    def _pressure_reward(self):
        return -self.get_pressure()

    def _weight_reward(self) :
        return self.get_pressure2()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _new_queue_reward(self) :
        return self.get_total_queued()

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        in_densities = self.get_state_density_in()
        #out_densities = self.get_state_density_out()
        in_queues = self.get_state_queue_in()
        #out_queues = self.get_state_queue_out()
        observation =  np.array(phase_id + min_green + in_densities + in_queues, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return abs(sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        ))

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    #pressure 2
    def get_tl_green_phases(self):
        logic = self.sumo.trafficlight.getAllProgramLogics(self.id)[0]
        #get only the green phases
        green_phases = [ p.state for p in logic.getPhases()
                         if 'y' not in p.state
                         and ('G' in p.state or 'g' in p.state) ]

        #sort to ensure parity between sims (for RL actions)
        return sorted(green_phases)

    def max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        green_phases = self.get_tl_green_phases()
        phase_lanes2 = self.phase_lanes(green_phases)
        max_pressure_lanes = {}
        for g in green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in phase_lanes2[g]:
                inc_lanes.add(l)
                for ol in netdata['lane'][l]['outgoing']:
                    out_lanes.add(ol)

            max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
        #print("mp_lanes :", str(max_pressure_lanes))
        return max_pressure_lanes

    def phase_lanes(self, actions):
        phase_lanes = {a:[] for a in actions}
        for a in actions:
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    green_lanes.add(netdata['inter'][self.id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(netdata['inter'][self.id]['tlsindex'][s])

            ###some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    def get_pressure2(self): #get weight = pressure/lane lenght
        mp_lanes = self.max_pressure_lanes() #inc & out lanes for each phase
        pressure = {}
        for phase in mp_lanes:
            in_out_lanes = mp_lanes[phase]

            inc_lanes = in_out_lanes['inc']
            veh_in_inc = sum((self.sumo.lane.getLastStepVehicleNumber(l)/self.sumo.lane.getLength(l)) for l in inc_lanes)

            out_lanes = in_out_lanes['out']
            veh_in_out = sum((self.sumo.lane.getLastStepVehicleNumber(k)/self.sumo.lane.getLength(k)) for k in out_lanes)

            pressure[phase] = abs(veh_in_inc - veh_in_out)/self.density_in_maxflow
        pressure_total = sum(pressure.values())
        return pressure_total

    def get_state_density_in(self):
        #mp_lanes = self.max_pressure_lanes() #inc & out lanes for each phase
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        density = {}
        for phase in self.mp_lanes:
            in_out_lanes = self.mp_lanes[phase]
            inc_lanes = in_out_lanes['inc']
            #get density
            density[phase] = math.ceil(sum([min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_lenght[lane] / vehicle_size_min_gap)) for lane in inc_lanes]))
        state_density = list(density.values())
        return state_density

    def get_state_density_out(self):
        #mp_lanes = self.max_pressure_lanes() #inc & out lanes for each phase
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        density = {}
        for phase in self.mp_lanes:
            in_out_lanes = self.mp_lanes[phase]
            out_lanes = in_out_lanes['out']
            #get density
            density[phase] = math.ceil(sum([min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.out_lanes_length[lane] / vehicle_size_min_gap)) for lane in out_lanes]))
        state_density = list(density.values())
        return state_density

    def get_state_queue_in(self): #incoming lanes
        #mp_lanes = self.max_pressure_lanes() #inc & out lanes for each phase
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        queue = {}
        for phase in self.mp_lanes:
            in_out_lanes = self.mp_lanes[phase]
            inc_lanes = in_out_lanes['inc']
            #get queue
            queue[phase] = math.ceil(sum([min(1, self.sumo.lane.getLastStepHaltingNumber(lane) / (self.lanes_lenght[lane] / vehicle_size_min_gap)) for lane in inc_lanes]))
        state_queue = list(queue.values())
        return state_queue

    def get_state_queue_out(self) :
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        queue = {}
        for phase in self.mp_lanes:
            in_out_lanes = self.mp_lanes[phase]
            out_lanes = in_out_lanes['out']
            #get queue
            queue[phase] = math.ceil(sum([min(1, self.sumo.lane.getLastStepHaltingNumber(lane) / (self.out_lanes_length[lane] / vehicle_size_min_gap)) for lane in out_lanes]))
        state_queue = list(queue.values())
        return state_queue

    def get_network_density(self): #for reward calculation
        vehperKMxlength = []
        all_lane = self.sumo.lane.getIDList()
        network_length = sum([self.sumo.lane.getLength(e) for e in all_lane])/1000
        for lane in all_lane :
            lane_lengthKM = self.sumo.lane.getLength(lane)/1000
            vehperKM = self.sumo.lane.getLastStepVehicleNumber(lane)/lane_lengthKM
            vehperKMxlength.append(vehperKM*lane_lengthKM)
        density = sum(vehperKMxlength)/network_length
        return density

    def exponential_reward(self):
        density = self.get_network_density()
        pressure = self.get_pressure2()
        queue = self.get_total_queued()
        alpha = 0.0277
        reward = -(pressure*(math.exp(-alpha*density)) + queue*(1-math.exp(-alpha*density)))
        return reward


    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }
