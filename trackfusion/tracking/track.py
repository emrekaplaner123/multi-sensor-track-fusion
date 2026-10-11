from dataclasses import dataclass
from enum import Enum

import numpy as np

from trackfusion.filtering.ekf import ExtendedKalmanFilter
from trackfusion.sensors.measurement import Measurement


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


@dataclass
class Track:
    track_id: int
    ekf: ExtendedKalmanFilter
    created_at: float
    last_time: float
    hits: int = 1
    misses: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE

    @property
    def state(self) -> np.ndarray:
        return self.ekf.x

    @property
    def covariance(self) -> np.ndarray:
        return self.ekf.P

    @property
    def position(self) -> np.ndarray:
        return self.ekf.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.ekf.x[2:]

    def predict_to(self, timestamp: float):
        dt = timestamp - self.last_time

        if dt < 0:
            raise ValueError("Cannot predict backwards in time.")

        if dt > 0:
            self.ekf.predict(dt)
            self.last_time = timestamp

    def update(
        self,
        measurement: Measurement,
        sensor
    ):
        self.ekf.update(
            z=measurement.z,
            h=sensor.h,
            jacobian=sensor.jacobian,
            R=measurement.R,
            angle_indices=(1,)
        )

        self.hits += 1
        self.misses = 0

    def mark_missed(self):
        self.misses += 1

    def confirm(self):
        self.status = TrackStatus.CONFIRMED