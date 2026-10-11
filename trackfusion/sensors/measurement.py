from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class MeasurementType(Enum):
    RANGE_BEARING = "range_bearing"
    RANGE_BEARING_DOPPLER = "range_bearing_doppler"


@dataclass
class Measurement:
    timestamp: float
    sensor_id: str
    z: np.ndarray
    R: np.ndarray
    measurement_type: MeasurementType
    truth_target_id: Optional[int] = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.R = np.asarray(self.R, dtype=float)

        if self.z.ndim != 1:
            raise ValueError("z must be one-dimensional.")

        if self.R.shape != (len(self.z), len(self.z)):
            raise ValueError(
                f"R must have shape {(len(self.z), len(self.z))}."
            )