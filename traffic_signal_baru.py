import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gym import spaces
import math
from collections import deque
from copy import deepcopy

from networkdata import NetworkData
nd = NetworkData('net_jkt-new/jkt-new.net.xml')
netdata = nd.get_net_data()

class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """
    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, begin_time,sumo):
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
        self.sumo = sumo

        self.density_in_maxflow = 1#veh/km

        self.build_phases()

        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes)) #remove duplicates
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes}
        self.out_lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.out_lanes} # out coming lanes

        self.mp_lanes = self.max_pressure_lanes()

        #self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+1+2*len(self.lanes), dtype=np.float32), high=np.ones(self.num_green_phases+1+2*len(self.lanes), dtype=np.float32))
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+1+2*(self.num_green_phases), dtype=np.float32), high=np.ones(self.num_green_phases+1+2*(self.num_green_phases), dtype=np.float32))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                       # Green Phase
            spaces.Discrete(2),                                           # Binary variable active if min_green seconds already elapsed
            *(spaces.Discrete(10) for _ in range(2*(self.num_green_phases)))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

    def build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases)//2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i,j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            #self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        in_densities = self.get_state_density_in()
        #out_densities = self.get_state_density_out()
        in_queues = self.get_state_queue_in()
        #out_queues = self.get_state_queue_out()
        observe = np.array(phase_id + min_green + in_densities + in_queues, dtype=np.float32)
        return observe

    def compute_reward(self):
        #self.last_reward = self._waiting_time_reward()
        #self.last_reward = self._pressure_reward()
        #self.last_reward = -(0.5*self._weight_reward() + 0.5*self._new_queue_reward())
        self.last_reward = self.exponential_reward()
        return self.last_reward

    def _pressure_reward(self):
        return -self.get_pressure()

    def _weight_reward(self) :
        return self.get_pressure2()

    def _new_queue_reward(self) :
        return self.get_total_queued()

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num()))**2

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
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

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
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
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_lenght[lane] / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 3 + 2.5  # 3(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepHaltingNumber(lane) / (self.lanes_lenght[lane] / vehicle_size_min_gap)) for lane in self.lanes]

    def get_total_queued(self):
        return (sum((self.sumo.lane.getLastStepHaltingNumber(lane)/self.sumo.lane.getLength(lane) for lane in self.lanes)))/self.density_in_maxflow


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
