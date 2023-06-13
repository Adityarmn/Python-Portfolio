"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gym import spaces
from environment.traffic_signal_v2 import TrafficSignal

#ts = traffic signal

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        in_densities = self.ts.get_state_density_in()
        #out_densities = self.get_state_density_out()
        in_queues = self.ts.get_state_queue_in()
        #out_queues = self.get_state_queue_out()
        observation =  np.array(phase_id + min_green + in_densities + in_queues, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * (self.ts.num_green_phases), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * (self.ts.num_green_phases), dtype=np.float32),
        )
