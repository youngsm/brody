x`from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import chroma

@dataclass
class Detector(ABC):
    
    detector_positions: np.ndarray
    detector_directions: np.ndarray
    num_pmt_types: int = 1
    
    @abstractmethod
    def plot(self, ev: chroma.Event, *args, **kwargs) -> None:
        """Plots the detector hits on a 2D plot."""
        
class TheiaDetector(Detector):
    def plot(self, ev: chroma.Event, *args, **kwargs) -> None:
        """Plots the detector hits on a 2D plot."""
        
    